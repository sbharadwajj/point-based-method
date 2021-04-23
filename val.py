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

network = PointNetCls_8k(feature_transform=False)
if opt.cuda:
    network.cuda()

network.apply(weights_init)

if opt.model != '':
    network.load_state_dict(torch.load(opt.model, map_location=torch.device('cpu')))
    print("Previous weight loaded ")

network.eval()

# inp_val = '/home/bharadwaj/dataset/final_training/val_partial'
# gt_val = '/home/bharadwaj/dataset/final_training/val_gt_npy'

# pose = '/home/bharadwaj/dataset/KITTI-360/data_poses'
# pose_dict = {}
# poses = os.listdir(pose)
# pose_folders = [os.path.join('/home/bharadwaj/dataset/KITTI-360/data_poses', folder) for folder in poses]
# pose_dict = {path.split("/")[-1]:np.loadtxt(path+"/poses.txt") for path in pose_folders}

partial_dir = '/home/bharadwaj/implementations/DATA/downsampled_inp/downsampled_predict/'
gt_dir = '/home/bharadwaj/implementations/DATA/downsampled_fused/downsampled_predict/'
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

def read_pcd(filename, center, dat=False):
    '''
    reads pcd, converts to float and normalizes
    '''
    if dat:
        point_set = (np.loadtxt(filename)).astype(np.float64)
    else:
        point_set = np.load(filename).astype(np.float64)
    point_set = point_set - center
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
    pcd = point_set / dist #scale
    return (torch.from_numpy(np.array(pcd)).float())

# Y_val = np.asarray(os.listdir(gt_val))
# Y_val = ["2013_05_28_drive_0010_sync_002756_002920_2920.npy", "2013_05_28_drive_0005_sync_004771_005011_4896.npy", "2013_05_28_drive_0005_sync_004998_005335_5022.npy", \
#     "2013_05_28_drive_0010_sync_002024_002177_2140.npy", "2013_05_28_drive_0009_sync_013701_013838_13832.npy", "2013_05_28_drive_0004_sync_003570_003975_3594.npy", \
#     "2013_05_28_drive_0003_sync_000002_000282_218.npy", "2013_05_28_drive_0009_sync_005422_005732_5424.npy", \
#     "2013_05_28_drive_0004_sync_002897_003118_2950.npy", "2013_05_28_drive_0009_sync_005422_005732_5626.npy"]
data_list = ['808.dat', "914.dat", "996.dat", "850.dat", "956.dat"]
with torch.no_grad():
    print(network)
    partial_list = []
    gt_list = []
    poses = []

    count = 0
    # for i in Y_val:
    #     split_list = i.split("_")
    #     file_name = split_list[-1].split(".")[0] + ".dat"
    #     drive_name =  "_".join(split_list[:6])
    #     x_path = os.path.join('_'.join(split_list[:-1]), file_name)
    
    #     center = trans_vector(file_name, pose_dict[drive_name]).transpose()

    #     partial = resample_pcd(read_pcd(os.path.join(inp_val, x_path), center, dat=True), 1024).unsqueeze(0)
    #     gt = resample_pcd(read_pcd(os.path.join(gt_val, i), center, dat=False), 5000).unsqueeze(0)
    #     partial_list.append(partial)
    #     gt_list.append(gt)
    #     poses.append(center)

    #     count +=1
    #     if count > 8:
    #         break
    for i in data_list:
        center = trans_vector(i, pose_matrix).transpose()
        partial = resample_pcd(read_pcd(partial_dir + i, center, dat=True), 2048).unsqueeze(0)
        gt = resample_pcd(read_pcd(gt_dir + i, center, dat=True), 8192).unsqueeze(0)
        partial_list.append(partial)
        gt_list.append(gt)
        poses.append(center)
    partial = torch.cat(partial_list, 0)
    gt = torch.cat(gt_list, 0)
    pose_mat = np.concatenate(poses, 0)
    pred, _, _ = network(partial.transpose(2,1).contiguous())
    np.savez('8k-nw39.npz', predictions=pred.numpy(), data=partial.numpy(), gt=gt.numpy(), poses=pose_mat)