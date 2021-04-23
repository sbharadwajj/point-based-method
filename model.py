from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 3072)
        self.fc3 = nn.Linear(3072, 2048*3)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(3072)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = torch.tanh(self.fc3(x))
        x = x.view(-1, 2048, 3)
        return x, trans, trans_feat
        
# ONLY FCN NETWORK
class PointNetCls_4(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls_4k, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 4096*2)
        self.fc4 = nn.Linear(4096*2, 4096*3)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(4096)
        self.bn3 = nn.BatchNorm1d(4096*2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_inp = x
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = F.relu(self.bn3(self.fc3(x)))
        x = torch.tanh(self.fc4(x))
        x = x.view(-1, 4096, 3)
        return x, trans, trans_feat

class PointNetCls_8k(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls_8k, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 8192)
        self.fc4 = nn.Linear(8192, 8192*3)
        # self.fc5 = nn.Linear(8192*2, 8192*3)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(4096)
        self.bn3 = nn.BatchNorm1d(4096*2)
        # self.bn4 = nn.BatchNorm1d(8192*2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_inp = x
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = F.relu(self.bn3(self.fc3(x)))
        # x = F.relu(self.bn4(self.fc4(x)))
        x = torch.tanh(self.fc4(x))
        x = x.view(-1, 8192, 3)
        return x, trans, trans_feat


# # WITH 1D DECONV
# class PointNetDeconvCls(nn.Module):
#     def __init__(self, k=2, feature_transform=False):
#         super(PointNetDeconvCls, self).__init__()
#         self.feature_transform = feature_transform
#         self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
#         self.decode = PointNetDeconvDecoder()
#         self.fc1 = nn.Linear(1024, 2048)
#         self.fc2 = nn.Linear(2048, 3072)
#         self.fc3 = nn.Linear(3072, 512*3)
#         self.dropout = nn.Dropout(p=0.3)
#         self.bn1 = nn.BatchNorm1d(2048)
#         self.bn2 = nn.BatchNorm1d(3072)
#         self.relu = nn.ReLU()
#         # self.tanh = nn.Tanh()


#     def forward(self, x):
#         global_feat, trans, trans_feat = self.feat(x)
#         x_additional = F.relu(self.bn1(self.fc1(global_feat)))
#         x_additional = F.relu(self.bn2(self.dropout(self.fc2(x_additional))))
#         x_additional = torch.tanh(self.fc3(x_additional))
#         x_additional = x_additional.view(-1, 512, 3)
#         x = self.decode(global_feat.view(-1, 4, 256))
#         x = torch.cat((x_additional, x.transpose(2,1)), dim=1)
#         return x, trans, trans_feat

# class PointNetDeconvDecoder(nn.Module):
#     def __init__(self):
#         super(PointNetDeconvDecoder, self).__init__()
#         self.convt1 = torch.nn.ConvTranspose1d(4, 64, 1)
#         self.conv1 = torch.nn.Conv1d(64, 64, 1)
#         self.convt2 = torch.nn.ConvTranspose1d(64, 128, 5, 2)
#         self.conv2 = torch.nn.Conv1d(128, 128, 1)
#         self.convt3 = torch.nn.ConvTranspose1d(128, 64, 5, 2)
#         self.conv3 = torch.nn.Conv1d(64, 64, 1)
#         self.convt4 = torch.nn.ConvTranspose1d(64, 3, 5, 2)
#         self.conv4 = torch.nn.Conv1d(3, 3, 1)
#         self.relu = nn.ReLU()
#         # self.tanh = nn.Tanh()

#     def forward(self, x):
#         x = self.convt1(x)
#         x = F.relu(self.conv1(x))
#         x = self.convt2(x)
#         x = F.relu(self.conv2(x))
#         x = self.convt3(x)
#         x = F.relu(self.conv3(x))
#         x = self.convt4(x)
#         x = torch.tanh(self.conv4(x))
#         return x



if __name__ == '__main__':
    data = Variable(torch.rand(8, 4, 32))
    deconv = PointNetDeconvDecoder()
    out = deconv(data)

    data_1 = Variable(torch.rand(8, 3, 1024))
    deconv_cls = PointNetDeconvCls()
    out_1 = deconv_cls(data_1)

    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())