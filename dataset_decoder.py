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
            self.inp = '/home/bharadwaj/implementations/DATA/downsampled_inp/downsampled_train'
            self.gt = '/home/bharadwaj/implementations/DATA/downsampled_fused/downsampled_train'
            self.Y = ["106.dat", "22.dat", "269.dat", "370.dat", "429.dat", "555.dat", "670.dat", "750.dat"]
            self.X = {"106.dat":0, "22.dat":1, "269.dat":2, "370.dat":3, "429.dat":4, "555.dat":5, "670.dat":6, "750.dat":7}
            random.shuffle(self.Y)
            self.len = len(self.Y)

        else:
            self.inp_val = '/home/bharadwaj/implementations/DATA/downsampled_inp/downsampled_train'
            self.gt_val = '/home/bharadwaj/implementations/DATA/downsampled_fused/downsampled_train'
            self.Y_val = ["106.dat", "22.dat", "269.dat", "370.dat", "429.dat", "555.dat", "670.dat", "750.dat"]
            self.X_val = {"106.dat":0, "22.dat":1, "269.dat":2, "370.dat":3, "429.dat":4, "555.dat":5, "670.dat":6, "750.dat":7}
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

        def trans_vector(model_id, poses):
            '''
            gets poses from pose.txt for each file
            '''
            id = float(model_id.split('.')[0])
            vec = np.squeeze(poses[poses[:,0] == id])
            reshaped = vec[1:].reshape(3,4)
            return reshaped[:,3:].astype(np.float64)

        def get_center(filename):
            '''
            returns center
            '''
            point_set = np.loadtxt(filename)
            center = np.expand_dims(np.mean(point_set, axis = 0), 0) # center
            return center

        def read_pcd(filename, center):
            '''
            reads pcd, converts to float and normalizes
            '''
            point_set = np.loadtxt(filename)
            point_set = point_set - center
            dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
            pcd = point_set / dist #scale
            return (torch.from_numpy(np.array(pcd)).float())
        

        one_hot = torch.zeros(1, self.len)
        center = trans_vector(model_id, self.pose_matrix).transpose()
        if self.train:
            # center = get_center(os.path.join(self.inp, model_id))
            # partial =read_pcd(os.path.join(self.inp, model_id), center)
            one_hot[torch.arange(1), self.X[model_id]] = 1
            complete = read_pcd(os.path.join(self.gt, model_id), center)
        else:
            # center = get_center(os.path.join(self.inp_val, model_id))
            # partial =read_pcd(os.path.join(self.inp_val, model_id), center)
            one_hot[torch.arange(1), self.Y[model_id]] = 1
            complete = read_pcd(os.path.join(self.gt_val, model_id), center)            
        return model_id, one_hot, resample_pcd(complete, self.npoints)

    def __len__(self):
        return self.len