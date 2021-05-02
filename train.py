# import open3d as o3d
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import sys
from dataset import *
from model import *
from utils import *
import os
import json
import time, datetime
from time import time
from pytorch3d.loss import chamfer_distance
# sys.path.append('/home/bharadwaj/implementations/baseline1-torch')
# from chamfer_distance import ChamferDistance

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--dataset', type=str, default = "kitti360"   ,  help='[kitti360, shapenet]')
parser.add_argument('--dataset_path', type=str, default = "shapenet/"   ,  help='dataset path')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--model', type=str, default = '',  help='optional reload model path')
parser.add_argument('--num_points', type=int, default = 5000,  help='number of points')
parser.add_argument('--num_point_partial', type=int, default = 1024,  help='number of points')
parser.add_argument('--n_primitives', type=int, default = 16,  help='number of surface elements')
parser.add_argument('--env', type=str, default ="KITTI360"   ,  help='visdom environment')
parser.add_argument('--cuda', type=bool, default = False   ,  help='if running on cuda')
parser.add_argument('--featTransform', type=bool, default = False   ,  help='if using feature transform')
parser.add_argument('--message', type=str, default = "training"   ,  help='specs of nw')


opt = parser.parse_args()
print (opt)

now = datetime.datetime.now()
save_path = 'overfit-feat-transform-regularizer-TRUE' + now.isoformat()
if not os.path.exists('./overfit_kitti360/'):
    os.mkdir('./overfit_kitti360/')
dir_name =  os.path.join('overfit_kitti360', save_path)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
logname = os.path.join(dir_name, 'overfit_kitti360.txt')
os.system('cp ./train.py %s' % dir_name)
os.system('cp ./dataset.py %s' % dir_name)
os.system('cp ./model.py %s' % dir_name)

if opt.dataset == 'kitti360':
    dataset = Kitti360(dataset_path=opt.dataset_path, train=True, npoints_partial = opt.num_point_partial, npoints=opt.num_points)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers), drop_last=True)
    dataset_test = Kitti360(opt.dataset_path, train=False, npoints_partial = opt.num_point_partial, npoints=opt.num_points)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                            shuffle=False, num_workers=int(opt.workers), drop_last=True)
elif opt.dataset == 'shapenet':
    dataset = Shapenet_allCategories(dataset_path=opt.dataset_path, train=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers))
    dataset_test = Shapenet_allCategories(dataset_path=opt.dataset_path, train=False, inp_points=opt.num_point_partial, npoints=opt.num_points)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                            shuffle=False, num_workers=int(opt.workers))

len_dataset = len(dataset)
print("Train Set Size: ", len_dataset)

# networks
if opt.num_points != 8192:
    network = PointNetCls(feature_transform=opt.featTransform)
else:
    network = PointNetCls_8k(feature_transform=opt.featTransform)

if opt.cuda:
    network.cuda()
network.apply(weights_init) #initialization of the weight

if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print("Previous weight loaded ")

# optimizer
lrate = 0.0001 #learning rate
optimizer = optim.Adam(network.parameters(), lr = lrate, weight_decay=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
writer = SummaryWriter(os.path.join('overfit_kitti360', save_path, 'logs'))

# chamferDist = ChamferDistance()
train_loss = AverageValueMeter()
val_loss = AverageValueMeter()
with open(logname, 'a') as f: #open and append
        f.write(str(network) + '\n')
        f.write(opt.message)

train_curve = []
val_curve = []


for epoch in range(opt.nepoch):
    #TRAIN MODE
    train_loss.reset()
    network.train()

    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()
        id, input, gt = data

        if opt.cuda:
            input = input.cuda()
            gt = gt.cuda()

        input = input.transpose(2,1) #.contiguous()
        pred, global_feat, trans_feat = network(input)
        loss, dist2 = chamfer_distance(gt, pred)
        loss_nodecay = (loss/2.0) * 1000

        if opt.featTransform:
            loss_net = loss_nodecay + feature_transform_regularizer(trans_feat) * 0.001
            '''
            used from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/ba4c5cf3a1e2b735d696c59a43c38345c3d003b9/models/pointnet_cls.py#L39
            '''
        else:
            loss_net = loss_nodecay
        loss_net.backward()
        # dist1, dist2 = chamferDist(gt, pred)
        # loss_net = ((torch.mean(dist1)) + (torch.mean(dist2)))/opt.batchSize
        # loss_net.backward()
        if opt.cuda:
            loss_item = loss_net.detach().cpu().item()
        else:  
            loss_item = loss_net.detach().item() #.cpu()
        train_loss.update(loss_item) 
        optimizer.step() 
        print(opt.env + ' train [%d: %d/%d]  trainloss: %f' %(epoch, i, len_dataset/opt.batchSize, loss_item))
    
    writer.add_scalar('train/loss', train_loss.avg , epoch)
    train_curve.append(train_loss.avg*100)

    # VALIDATION
    if epoch % 5 == 0:
        val_loss.reset()
        network.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader_test, 0):
                id, input, gt = data
                if opt.cuda:
                    input = input.cuda()
                    gt = gt.cuda()

                input = input.transpose(2,1) #.contiguous()
                pred, _, trans_feat = network(input)  
                loss_nodecay, _ = chamfer_distance(gt, pred)  
                # dist1, dist2 = chamferDist(gt, pred)
                # loss_net = ((torch.mean(dist1)) + (torch.mean(dist2)))/opt.batchSize
                loss_nodecay = (loss/2.0) * 1000

                if opt.featTransform:
                    loss_net = loss_nodecay + feature_transform_regularizer(trans_feat) * 0.001
                    '''
                    used from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/ba4c5cf3a1e2b735d696c59a43c38345c3d003b9/models/pointnet_cls.py#L39
                    '''
                else:
                    loss_net = loss_nodecay
                if opt.cuda:     
                    val_loss_item = loss_net.detach().cpu().item() #.cpu()    
                else:
                    val_loss_item = loss_net.detach().item()      
                val_loss.update(val_loss_item) 
                idx = random.randint(0, input.size()[0] - 1)
                print(opt.env + ' val [%d: %d/%d]  emd1: %f' %(epoch, i, len_dataset/opt.batchSize, val_loss_item))

    val_curve.append(val_loss.avg)
    writer.add_scalar('val/loss', val_loss.avg*100, epoch)

    log_table = {
      "train_loss" : train_loss.avg*100,
      "val_loss" : val_loss.avg*100,
      "epoch" : epoch,
      "lr" : lrate,

    }
    with open(logname, 'a') as f: 
        f.write('json_stats: ' + json.dumps(log_table) + '\n')

    scheduler.step()
    print('saving net...')
    if epoch % 50 == 0:
        torch.save(network.state_dict(), '%s/network_%d.pth' % (dir_name, epoch))
