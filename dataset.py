# import open3d as o3d
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import random
#from utils import *

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]
           
class Kitti360(data.Dataset): 
    def __init__(self, train = True, npoints = 8192):
        if train:
            self.inp = '/home/bharadwaj/dataset/train_partial'
            self.gt = '/home/bharadwaj/dataset/train_gt_npy'
            X = np.asarray(os.listdir(self.inp))
            self.Y = np.asarray(os.listdir(self.gt))
            self.len = len(self.Y)

        else:
            self.inp_val = '/home/bharadwaj/dataset/val_partial'
            self.gt_val = '/home/bharadwaj/dataset/val_gt_npy'
            X_val = np.asarray(os.listdir(self.inp_val))
            self.Y_val = np.asarray(os.listdir(self.gt_val))
            self.len = len(self.Y_val)
        self.npoints = npoints
        self.train = train
        self.pose = '/home/bharadwaj/implementations/DATA/poses.txt'
        self.pose_matrix = np.loadtxt(self.pose)

    def __getitem__(self, index):
        if self.train:
            model_id = self.Y[index]  
        else:
            model_id = self.Y_val[index]      

        # print(os.path.join(self.inp, model_id))
        split_list = model_id[index].split("_")
        x_path = os.path.join('_'.join(split_list[:-1]), split_list[-1].split(".")[0] + ".dat")
        def trans_vector(model_id, poses):
            '''
            gets poses from pose.txt for each file
            '''
            id = float(model_id.split('.')[0])
            vec = np.squeeze(poses[poses[:,0] == id])
            reshaped = vec[1:].reshape(3,4)
            return reshaped[:,3:].astype(np.float64)

        def read_pcd(filename, center):
            '''
            reads pcd, converts to float and normalizes
            '''
            point_set = np.load(filename)
            point_set = point_set - center
            dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
            pcd = point_set / dist #scale
            return (torch.from_numpy(np.array(pcd)).float())
        
        center = trans_vector(model_id, self.pose_matrix).transpose()
        if self.train:
            partial =read_pcd(os.path.join(self.inp, x_path), center)
            complete = read_pcd(os.path.join(self.gt, model_id), center)
        else:
            partial =read_pcd(os.path.join(self.inp_val, x_path), center)
            complete = read_pcd(os.path.join(self.gt_val, model_id), center)            
        return model_id, resample_pcd(partial, 1024), resample_pcd(complete, self.npoints)

    def __len__(self):
        return self.len