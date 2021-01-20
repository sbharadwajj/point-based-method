# import open3d as o3d
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import sys
from dataset import *
from model import *
from utils import *
import os
import json
import time, datetime
import visdom
from time import time
# from chamferdist import ChamferDistance
# sys.path.append('/home/bharadwaj/implementations/baseline1-torch')
# from chamfer_distance import ChamferDistance

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--model', type=str, default = '',  help='optional reload model path')
parser.add_argument('--num_points', type=int, default = 8192,  help='number of points')
parser.add_argument('--n_primitives', type=int, default = 16,  help='number of surface elements')
parser.add_argument('--env', type=str, default ="MSN_TRAIN"   ,  help='visdom environment')
parser.add_argument('--cuda', type=bool, default = False   ,  help='if running on cuda')
parser.add_argument('--fc_nw', type=bool, default = True   ,  help='running vanilla')
parser.add_argument('--message', type=str, default = "training"   ,  help='specs of nw')

opt = parser.parse_args()
print (opt)

# create paths
# vis = visdom.Visdom(port = 8097, env=opt.env) # set your port
now = datetime.datetime.now()
save_path = now.isoformat()
if not os.path.exists('./log/'):
    os.mkdir('./log/')
dir_name =  os.path.join('log', save_path)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
logname = os.path.join(dir_name, 'log.txt')
os.system('cp ./train.py %s' % dir_name)
os.system('cp ./dataset.py %s' % dir_name)
os.system('cp ./model.py %s' % dir_name)

opt.manualSeed = random.randint(1, 10000) 
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
best_val_loss = 10

# dataloader
dataset = Kitti360(train=True, npoints=opt.num_points)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))
dataset_test = Kitti360(train=False, npoints=opt.num_points)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                          shuffle=False, num_workers=int(opt.workers))

len_dataset = len(dataset)
print("Train Set Size: ", len_dataset)

# networks
print(opt.fc_nw)
if opt.fc_nw:
    network = PointNetCls(feature_transform=True)
else:
    print("Deconv Network")
    network = PointNetDeconvCls()

if opt.cuda:
    network.cuda()
network.apply(weights_init) #initialization of the weight

if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print("Previous weight loaded ")

# optimizer
lrate = 0.0001 #learning rate
optimizer = optim.Adam(network.parameters(), lr = lrate)

entropy = nn.CrossEntropyLoss()
chamferDist = ChamferDistance()
train_loss = AverageValueMeter()
val_loss = AverageValueMeter()
with open(logname, 'a') as f: #open and append
        f.write(str(network) + '\n')
        f.write(opt.message)


train_curve = []
val_curve = []
labels_generated_points = torch.Tensor(range(1, (opt.n_primitives+1)*(opt.num_points//opt.n_primitives)+1)).view(opt.num_points//opt.n_primitives,(opt.n_primitives+1)).transpose(0,1)
labels_generated_points = (labels_generated_points)%(opt.n_primitives+1)
labels_generated_points = labels_generated_points.contiguous().view(-1)


for epoch in range(opt.nepoch):
    #TRAIN MODE
    train_loss.reset()
    network.train()
    
    # learning rate schedule
    if epoch==20:
        optimizer = optim.Adam(network.parameters(), lr = lrate/10.0)
    if epoch==40:
        optimizer = optim.Adam(network.parameters(), lr = lrate/100.0)

    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()
        id, input, gt = data
        input = input.float()#.cuda()
        gt = gt.squeeze()
        if opt.cuda:
            input = input.cuda()
            gt = gt.cuda()

        input = input.transpose(2,1).contiguous()
        pred = network(input)

        loss_net = entropy(pred, gt)
        loss_net.backward()
        if opt.cuda:
            loss_item = loss_net.detach().cpu().item()
        else:  
            loss_item = loss_net.detach().item() #.cpu()
        train_loss.update(loss_item) 
        optimizer.step() 

        print(opt.env + ' train [%d: %d/%d]  trainloss: %f' %(epoch, i, len_dataset/opt.batchSize, loss_item))
    train_curve.append(train_loss.avg)

    # VALIDATION
    if epoch % 5 == 0:
        val_loss.reset()
        network.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader_test, 0):
                id, input, gt = data
                input = input.float()#.cuda()
                gt = gt.squeeze()

                if opt.cuda:
                    input = input.cuda()
                    gt = gt.cuda()

                input = input.transpose(2,1).contiguous()
                pred = network(input)    
                loss_net = entropy(pred, gt)

                if opt.cuda:     
                    val_loss_item = loss_net.detach().cpu().item() #.cpu()    
                else:
                    val_loss_item = loss_net.detach().item()      
                val_loss.update(val_loss_item) 
                idx = random.randint(0, input.size()[0] - 1)
                print(opt.env + ' val [%d: %d/%d]  emd1: %f' %(epoch, i, len_dataset/opt.batchSize, val_loss_item))

    val_curve.append(val_loss.avg)


    log_table = {
      "train_loss" : train_loss.avg,
      "val_loss" : val_loss.avg,
      "epoch" : epoch,
      "lr" : lrate,
      "bestval" : best_val_loss,

    }
    with open(logname, 'a') as f: 
        f.write('json_stats: ' + json.dumps(log_table) + '\n')

    print('saving net...')
    torch.save(network.state_dict(), '%s/network_%d.pth' % (dir_name, epoch))
