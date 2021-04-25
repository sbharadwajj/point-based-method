import sys
# import open3d as o3d
from model import *
from utils import *
import argparse
import random
import numpy as np
import torch
import os
from dataset import *
from pytorch3d.loss import chamfer_distance

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=8, help='partial batch size')
parser.add_argument('--dataset_path', type=str, default = "shapenet/"   ,  help='dataset path')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--model', type=str, default = '',  help='optional reload model path')
parser.add_argument('--num_points', type=int, default = 2048,  help='number of points')
parser.add_argument('--n_primitives', type=int, default = 16,  help='number of surface elements')
parser.add_argument('--env', type=str, default ="MSN_TRAIN"   ,  help='visdom environment')
parser.add_argument('--cuda', type=bool, default = False   ,  help='if running on cuda')
parser.add_argument('--fc_nw', type=bool, default = True   ,  help='running vanilla')
parser.add_argument('--message', type=str, default = "training"   ,  help='specs of nw')


opt = parser.parse_args()
print (opt)

network = PointNetCls(feature_transform=False)
if opt.cuda:
    network.cuda()

network.apply(weights_init)

if opt.model != '':
    network.load_state_dict(torch.load(opt.model, map_location=torch.device('cpu')))
    print("Previous weight loaded ")

network.eval()

# dataset = Shapenet(dataset_path=opt.dataset_path, train=True, inp_points = 1024, npoints=opt.num_points)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
#                                           shuffle=True, num_workers=int(opt.workers))
# dataset_test = Shapenet(dataset_path=opt.dataset_path, train=False, inp_points=1024, npoints=opt.num_points)
# dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
#                                           shuffle=False, num_workers=int(opt.workers))

dataset = Shapenet_allCategories(dataset_path=opt.dataset_path, train=True, inp_points = 1024, npoints=opt.num_points)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))
dataset_test = Shapenet_allCategories(dataset_path=opt.dataset_path, train=False, inp_points=1024, npoints=opt.num_points)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                          shuffle=False, num_workers=int(opt.workers))

with torch.no_grad():
    print(network)
    partial_list = []
    gt_list = []
    pred_list = []
    
    for i, data in enumerate(dataloader_test, 0):
        id, partial, gt = data
        partial = partial.float()#.cuda()
        gt = gt.float()#.cuda()

        if opt.cuda:
            partial = partial.cuda()
            gt = gt.cuda()

        # partial = partial.transpose(2,1).contiguous()
        pred, _, _ = network(partial)  
        loss_net, _ = chamfer_distance(pred, gt)  

        partial_list.append(partial)
        gt_list.append(gt)
        pred_list.append(pred)

        
    partial = torch.cat(partial_list, 0)
    gt = torch.cat(gt_list, 0)
    pred = torch.cat(pred_list, 0)
    np.savez('allcat-epoch75.npz', predictions=pred.numpy(), data=partial.numpy(), gt=gt.numpy())