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

network = PointNetCls(feature_transform=True)
# network = PointNetDeconvCls()
# import pdb; pdb.set_trace()
if opt.cuda:
    network.cuda()

network.apply(weights_init)

# vis = visdom.Visdom(port = 8097, env=opt.env) # set your port

if opt.model != '':
    network.load_state_dict(torch.load(opt.model, map_location=torch.device('cpu')))
    print("Previous weight loaded ")

network.eval()
# with open(os.path.join('./data/val.list')) as file:
#     model_list = [line.strip().replace('/', '_') for line in file]

partial_dir = '/home/bharadwaj/implementations/DATA/downsampled_inp/downsampled_predict/'
gt_dir = '/home/bharadwaj/implementations/DATA/downsampled_fused/downsampled_predict/'
# vis = visdom.Visdom(port = 8097, env=opt.env) # set your port

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]


# labels_generated_points = torch.Tensor(range(1, (opt.n_primitives+1)*(opt.num_points//opt.n_primitives)+1)).view(opt.num_points//opt.n_primitives,(opt.n_primitives+1)).transpose(0,1)
# labels_generated_points = (labels_generated_points)%(opt.n_primitives+1)
# labels_generated_points = labels_generated_points.contiguous().view(-1)
def read_pcd(filename):
    point_set = np.loadtxt(filename)
    point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
    pcd = point_set / dist #scale
    return torch.from_numpy(np.array(pcd)).float()

# data_list = ["106.dat", "22.dat", "269.dat", "370.dat", "429.dat", "555.dat", "670.dat", "750.dat"]
# data_list = ['808.dat', "914.dat", "996.dat", "850.dat", "956.dat"]
data_list = ["9.dat"]
with torch.no_grad():
# for i, model in enumerate(model_list):
    print(network)
    partial_list = []
    gt_list = []
    for i in data_list:
        partial = resample_pcd(read_pcd(partial_dir + i), 1024).unsqueeze(0)
        gt = resample_pcd(read_pcd(gt_dir + i), 5000).unsqueeze(0)
        partial_list.append(partial)
        gt_list.append(gt)
    partial = torch.cat(partial_list, 0)
    gt = torch.cat(gt_list, 0)
    # import pdb; pdb.set_trace()


    # partial = torch.zeros((8, 1024, 3), device='cuda')
    # gt = torch.zeros((8, opt.num_points, 3), device='cuda')
    # for j in range(8):
    #     pcd = o3d.io.read_point_cloud(os.path.join(partial_dir, model + '_' + str(j) + '_denoised.pcd'))
    #     partial[j, :, :] = torch.from_numpy(resample_pcd(np.array(pcd.points), 5000))
    #     pcd = o3d.io.read_point_cloud(os.path.join(gt_dir, model + '.pcd'))
    #     gt[j, :, :] = torch.from_numpy(resample_pcd(np.array(pcd.points), opt.num_points))
    pred, _, _ = network(partial.transpose(2,1).contiguous())
    np.savez('singlefileDiffCenter49.npz', predictions=pred.numpy(), data=partial.numpy(), gt=gt.numpy())
    # dist, _ = EMD(output1, gt, 0.002, 10000)
    # emd1 = torch.sqrt(dist).mean()
    # dist, _ = EMD(output2, gt, 0.002, 10000)
    # emd2 = torch.sqrt(dist).mean()
    # idx = random.randint(0, 49)
    # vis.scatter(X = gt[idx].data.cpu(), win = 'GT',
    #             opts = dict(title = model, markersize = 2))
    # vis.scatter(X = partial[idx].data.cpu(), win = 'INPUT',
    #             opts = dict(title = model, markersize = 2))
    # vis.scatter(X = output1[idx].data.cpu(),
    #             Y = labels_generated_points[0:output1.size(1)],
    #             win = 'COARSE',
    #             opts = dict(title = model, markersize=2))
    # vis.scatter(X = output2[idx].data.cpu(),
    #             win = 'OUTPUT',
    #             opts = dict(title = model, markersize=2))
    # print(opt.env + ' val [%d/%d]  emd1: %f emd2: %f expansion_penalty: %f' %(i + 1, len(model_list), emd1.item(), emd2.item(), expansion_penalty.mean().item()))
