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
parser.add_argument('--dataset', type=str, default = "kitti360"   ,  help='dataset path')
parser.add_argument('--dataset_path', type=str, default = "shapenet/"   ,  help='dataset path')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--model', type=str, default = '',  help='optional reload model path')
parser.add_argument('--num_points', type=int, default = 2048,  help='number of points')
parser.add_argument('--num_point_partial', type=int, default = 1024,  help='number of points')
parser.add_argument('--n_primitives', type=int, default = 16,  help='number of surface elements')
parser.add_argument('--env', type=str, default ="MSN_TRAIN"   ,  help='visdom environment')
parser.add_argument('--cuda', type=bool, default = False   ,  help='if running on cuda')
parser.add_argument('--featTransform', type=bool, default = False   ,  help='if using feature transform')
parser.add_argument('--message', type=str, default = "training"   ,  help='specs of nw')


opt = parser.parse_args()
print (opt)

# networks
if opt.num_points != 8192:
    network = PointNetCls(feature_transform=opt.featTransform)
else:
    network = PointNetCls_8k(feature_transform=opt.featTransform)
if opt.cuda:
    network.cuda()

# network.apply(weights_init)

if opt.cuda:
    network.load_state_dict(torch.load(opt.model, map_location=torch.device('cuda')))
else:
    network.load_state_dict(torch.load(opt.model, map_location=torch.device('cpu')))
print("Previous weight loaded ")

network.eval()

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

print(len(dataset_test))
print(len(dataset))

with torch.no_grad():
    print(network)
    partial_list = []
    gt_list = []
    pred_list = []

    # global_feat_list = []
    
    val_loss = AverageValueMeter()
    for i, data in enumerate(dataloader, 0):
        id, partial, gt = data
        partial = partial #.float()#.cuda()
        gt = gt #.float()#.cuda()

        if opt.cuda:
            partial = partial.cuda()
            gt = gt.cuda()

        partial = partial.transpose(2,1) #.contiguous()
        pred, global_feat, _ = network(partial)  
        loss, _ = chamfer_distance(gt, pred)  
        loss_nodecay = (loss/2.0) * 1000

        if opt.featTransform:
            loss_net = loss_nodecay + feature_transform_regularizer(trans_feat) * 0.001
            '''
            used from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/ba4c5cf3a1e2b735d696c59a43c38345c3d003b9/models/pointnet_cls.py#L39
            '''
        else:
            loss_net = loss_nodecay

        partial_list.append(partial)
        gt_list.append(gt)
        pred_list.append(pred)
        # global_feat_list.append(global_feat)

        val_loss.update(loss_net.detach().cpu().item())


    print("val loss avg:",val_loss.avg*100) 
    partial = torch.cat(partial_list, 0)
    gt = torch.cat(gt_list, 0)
    pred = torch.cat(pred_list, 0)
    # global_feats = torch.cat(global_feat_list, 0)


    name = opt.model.split("/")[-1]
    np.savez(name.split(".")[0] + "lrSchedu.npz", predictions=pred.cpu().numpy(), data=partial.cpu().numpy(), gt=gt.cpu().numpy())