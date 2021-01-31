import sys
# import open3d as o3d
from model import *
from utils import *
import argparse
import random
import numpy as np
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = './trained_model/network.pth',  help='optional reload model path')
parser.add_argument('--num_points', type=int, default = 8192,  help='number of points')
parser.add_argument('--n_primitives', type=int, default = 16,  help='number of primitives in the atlas')
parser.add_argument('--env', type=str, default ="MSN_VAL"   ,  help='visdom environment') 
parser.add_argument('--cuda', type=bool, default = False   ,  help='if running on cuda')


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

partial_dir = '/home/bharadwaj/implementations/DATA/downsampled_inp/downsampled_train/'
gt_dir = '/home/bharadwaj/implementations/DATA/downsampled_fused/downsampled_train/'
pose = '/home/bharadwaj/implementations/DATA/poses.txt'
pose_matrix = np.loadtxt(pose)

def trans_vector(model_id, poses):
    '''
    gets poses from pose.txt for each file
    '''
    id = float(model_id.split('.')[0])
    vec = np.squeeze(poses[poses[:,0] == id])
    reshaped = vec[1:].reshape(3,4)
    return reshaped[:,3:].astype(np.float64)

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]

def read_pcd(filename, center):
    point_set = np.loadtxt(filename)
    point_set = point_set - center
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
    pcd = point_set / dist #scale
    return torch.from_numpy(np.array(pcd)).float()

data_list = ["106.dat", "22.dat", "269.dat", "370.dat", "429.dat", "555.dat", "670.dat", "750.dat"]
# data_list = ['808.dat', "914.dat", "996.dat", "850.dat", "956.dat"]
# data_list = ["9.dat"]
with torch.no_grad():
# for i, model in enumerate(model_list):
    print(network)
    partial_list = []
    gt_list = []
    poses = []
    for i in data_list:
        center = trans_vector(i, pose_matrix).transpose()
        partial = resample_pcd(read_pcd(partial_dir + i, center), 1024).unsqueeze(0)
        gt = resample_pcd(read_pcd(gt_dir + i, center), 5000).unsqueeze(0)
        partial_list.append(partial)
        gt_list.append(gt)
        poses.append(center)
    partial = torch.cat(partial_list, 0)
    gt = torch.cat(gt_list, 0)
    pose_mat = np.concatenate(poses, 0)
    pred, _, _ = network(partial.transpose(2,1).contiguous())
    np.savez('8-img-1999.npz', predictions=pred.numpy(), data=partial.numpy(), gt=gt.numpy(), poses=pose_mat)