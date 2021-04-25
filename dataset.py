# import open3d as o3d
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import random

from data_utils import load_h5, pad_cloudN, augment_cloud
#from utils import *

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]
           
class Kitti360(data.Dataset): 
    def __init__(self, dir_name, train = True, npoints_partial = 2048, npoints = 4096):
        if train:
            self.inp = '/home/bharadwaj/dataset/final_training/train_partial'
            self.gt = '/home/bharadwaj/dataset/final_training/train_gt_npy'
            X = np.asarray(os.listdir(self.inp))
            Y = np.asarray(os.listdir(self.gt))
            np.random.shuffle(Y)
            self.Y = Y
            self.len = len(self.Y)
            path = os.path.join(dir_name, "train_files.txt")
            np.savetxt(path, self.Y, fmt='%s')
        else:
            self.inp_val = '/home/bharadwaj/dataset/final_training/val_partial'
            self.gt_val = '/home/bharadwaj/dataset/final_training/val_gt_npy'
            X_val = np.asarray(os.listdir(self.inp_val))
            self.Y_val = np.asarray(os.listdir(self.gt_val))
            self.len = len(self.Y_val)
        self.npoints_partial = npoints_partial
        self.npoints = npoints
        self.train = train
        self.pose = '/home/bharadwaj/dataset/KITTI-360/data_poses'
        pose_dict = {}
        poses = os.listdir(self.pose)
        pose_folders = [os.path.join('/home/bharadwaj/dataset/KITTI-360/data_poses', folder) for folder in poses]
        self.pose_dict = {path.split("/")[-1]:np.loadtxt(path+"/poses.txt") for path in pose_folders}

    def trans_vector(self, model_id, poses):
        '''
        gets poses from pose.txt for each file
        '''
        id = float(model_id.split('.')[0])
        vec = np.squeeze(poses[poses[:,0] == id])
        reshaped = vec[1:].reshape(3,4)
        return reshaped[:,3:].astype(np.float64)

    def read_pcd(self, filename, center, dat=False):
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
        return pcd.astype(np.float)

    def __getitem__(self, index):
        if self.train:
            model_id = self.Y[index]  
        else:
            model_id = self.Y_val[index]      

        split_list = model_id.split("_")
        file_name = split_list[-1].split(".")[0] + ".dat"
        drive_name =  "_".join(split_list[:6])
        x_path = os.path.join('_'.join(split_list[:-1]), file_name)
        # print(model_id)
        center = self.trans_vector(file_name, self.pose_dict[drive_name]).transpose()
        
        if self.train:
            partial_ = self.read_pcd(os.path.join(self.inp, x_path), center, dat=True)
            complete_ = self.read_pcd(os.path.join(self.gt, model_id), center, dat=False)
            complete, partial = augment_cloud([complete_, partial_])
        else:
            partial = self.read_pcd(os.path.join(self.inp_val, x_path), center, dat=True)
            complete = self.read_pcd(os.path.join(self.gt_val, model_id), center, dat=False)            
        return model_id, resample_pcd(partial, self.npoints_partial), resample_pcd(complete, self.npoints)

    def __len__(self):
        return self.len

class Shapenet(data.Dataset): 
    def __init__(self, dataset_path, train = True, inp_points=1024 , npoints = 2048):
        self.train = train
        self.inp_points = inp_points
        self.npoints = npoints
        if self.train:
            # train on a single category to test
            self.inp = os.path.join(dataset_path, "train", "partial")
            self.gt = os.path.join(dataset_path, "train", "gt")
            self.X = os.listdir(self.inp)
            self.y = os.listdir(self.gt)
            self.len = len(os.listdir(self.inp))
        else:
            self.inp = os.path.join(dataset_path, "val", "partial")
            self.gt = os.path.join(dataset_path, "val", "gt")
            self.X = os.listdir(self.inp)
            self.y = os.listdir(self.gt)
            self.len = len(os.listdir(self.inp))
        self.npoints = npoints

    def get_pair(self, fname, train):
        partial = load_h5(os.path.join(self.inp, fname))
        gtpts = load_h5(os.path.join(self.gt, fname))
        partial  = pad_cloudN(partial, self.inp_points)
        return partial, gtpts

    def load_data(self, fname):
        pair = self.get_pair(fname, train=self.train == 'train')
        partial = pair[0].T
        target = pair[1]
        cloud_meta = ['{}.{:d}'.format('/'.join(fname.split('/')[-2:]),0),]
        return cloud_meta, partial, target

    def __getitem__(self, index):
        model_id = self.X[index]
        return self.load_data(model_id)
    
    def __len__(self):
        return self.len

class Shapenet_allCategories(data.Dataset): 
    def __init__(self, dataset_path, train = True, inp_points=1024 , npoints = 2048):
        self.train = train
        self.inp_points = inp_points
        self.npoints = npoints
        if self.train:
            self.inp = os.path.join(dataset_path, "train", "partial")
            self.gt = os.path.join(dataset_path, "train", "gt")
            self.data_paths = sorted([os.path.join(dataset_path, 'train', 'partial', k.rstrip()+ '.h5') for k in open(os.path.join(dataset_path, 'train.list')).readlines()])
            self.len = len(self.data_paths)
        else:
            self.inp = os.path.join(dataset_path, "val", "partial")
            self.gt = os.path.join(dataset_path, "val", "gt")
            self.data_paths = sorted([os.path.join(dataset_path, 'val', 'partial', k.rstrip()+ '.h5') for k in open(os.path.join(dataset_path, 'val.list')).readlines()])
            self.X = os.listdir(self.inp)
            self.y = os.listdir(self.gt)
            self.len = len(self.data_paths)
        self.npoints = npoints

    def get_pair(self, fname, train):
        folder = fname.split("/")[-2]
        name = fname.split("/")[-1]
        partial = load_h5(os.path.join(self.inp, folder, name))
        gtpts = load_h5(os.path.join(self.gt, folder, name))
        # FIX CODE AND ADD AUGMENTATION
        if train:
            gtpts, partial = augment_cloud([gtpts, partial])
        partial  = pad_cloudN(partial, self.inp_points)
        return partial, gtpts

    def load_data(self, fname):
        pair = self.get_pair(fname, train=self.train == 'train')
        partial = pair[0].T
        target = pair[1]
        cloud_meta = ['{}.{:d}'.format('/'.join(fname.split('/')[-2:]),0),]
        return cloud_meta, partial, target

    def __getitem__(self, index):
        model_id = self.data_paths[index]
        return self.load_data(model_id)
    
    def __len__(self):
        return self.len