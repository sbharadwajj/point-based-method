# import open3d as o3d
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import random

from data_utils import load_h5, pad_cloudN, augment_cloud
#from utils import *

           
class Kitti360(data.Dataset): 
    def __init__(self, dataset_path, train = True, weights = False, npoints_partial = 1024, npoints = 2048):
        self.train = train
        self.npoints = npoints
        self.weights = weights
        if self.train:
            self.inp = os.path.join(dataset_path, "train", "partial")
            self.gt = os.path.join(dataset_path, "train", "gt")
            self.X = os.listdir(self.inp)
            self.Y = os.listdir(self.gt)

            # sort_y = sorted(self.Y)[0::2000] # choose 10
            # self.Y = sort_y
            self.len = len(self.Y)
        else:
            self.inp = os.path.join(dataset_path, "val", "partial")
            self.gt = os.path.join(dataset_path, "val", "gt")
            self.X = os.listdir(self.inp)
            self.Y = os.listdir(self.gt)[:2000]
            # sort_y = sorted(self.Y)[0::2000] # choose the 100th one
            # print(sort_y)
            # self.Y = sort_y
            self.len = len(self.Y)

        # print(self.inp)
        # print(self.gt)
        '''
        loads poses to a dictonary to read
        '''
        self.pose = '/home/bharadwaj/dataset/KITTI-360/data_poses'
        pose_dict = {}
        poses = os.listdir(self.pose)
        pose_folders = [os.path.join('/home/bharadwaj/dataset/KITTI-360/data_poses', folder) for folder in poses]
        self.pose_dict = {path.split("/")[-1]:np.loadtxt(path+"/poses.txt") for path in pose_folders}

    def get_weight_vec(self, points_z, percent, array_pcd, axis):
        thresh = np.quantile(points_z, percent)
        bottom = array_pcd[:, axis] < thresh
        top = array_pcd[:, axis] > thresh
        weights_array = (np.ones((self.npoints)).astype(float)) * 2.0
        weights_array[bottom] = 1.0 # WEIGHTS FOR BOTTOM 60 %
        assert(weights_array[top] == 2.0).all()
        return weights_array

    def get_translation_vec(self, model_id, poses):
        '''
        gets poses from pose.txt for each file
        '''
        id = float(model_id)
        vec = np.squeeze(poses[poses[:,0] == id])
        reshaped = vec[1:].reshape(3,4)
        return reshaped[:,3:].astype(np.float64)

    def read_pcd(self, filename, center):
        '''
        reads pcd and normalizes
        '''
        point_set = np.load(filename) # .astype(np.float64) saved as float already
        point_set = point_set - center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        pcd = point_set / dist # scale
        return pcd #.astype(np.float)

    def __getitem__(self, index):        
        model_id = self.Y[index] 
        
        split_list = model_id.split("_") 
        file_name = split_list[-1].split(".")[0]
        drive_name =  "_".join(split_list[:6])
        center = self.get_translation_vec(file_name, self.pose_dict[drive_name]).transpose()

        partial = self.read_pcd(os.path.join(self.inp, model_id), center)
        complete = self.read_pcd(os.path.join(self.gt, model_id), center)        
        if self.train:
            complete, partial = augment_cloud([complete, partial]) 
        if self.weights:
            model_id = self.get_weight_vec(complete[:,2], 0.6, complete, axis=2) #z axis         
        return model_id, partial.astype(np.float32), complete.astype(np.float32)

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
            self.Y = os.listdir(self.gt)
            self.len = len(os.listdir(self.inp))
        else:
            self.inp = os.path.join(dataset_path, "val", "partial")
            self.gt = os.path.join(dataset_path, "val", "gt")
            self.X = os.listdir(self.inp)
            self.Y = os.listdir(self.gt)
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
            self.Y = os.listdir(self.gt)
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