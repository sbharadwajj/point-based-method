'''
run:
`python utils-avg/pointcloud/create_train_pair.py 2013_05_28_drive_train_only.txt path_fused_cloud save_folder radius pose_folder path_sparse_cloud save`

example:
`python utils-avg/pointcloud/create_train_pair.py 2013_05_28_drive_train_only.txt data_3d_semantics/ baseline_data/train_fused/ 70 data_poses/ train_sparse/ save`

my code
python3 create_train_pair.py 2013_05_28_drive_train_only.txt ../data_3d_semantics/ 4096-8192-kitti360/train 70 ../KITTI-360/data_poses/ ../final_training/train_partial/ save train_list.txt
'''

import open3d as o3d 
import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import sys
import os.path as osp
import h5py

voxel_size_gt = 1.3
voxel_size_gt_2 = 1.0
voxel_size_partial = 0.5
voxel_size_partial_2 = 0.3
n_gt = 8192
n_partial = 4096

def get_X_names(start, end):
    X_list = []
    for n in range((end-start)+1):
        X_list.append(start+n)
    assert(X_list[0] == start)
    assert(X_list[len(X_list)-1] == end)
    return X_list

def points_to_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)  
    return pcd

def crop_pointcloud(cloud, r, ref):
    '''
    crop pcd along with semantic labels
    '''
    points = np.asarray(cloud.points).astype(np.float64)
    dist = np.linalg.norm(np.transpose(ref) - points, axis=1)
    bounding_box = points[(dist < r)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(bounding_box)
    return pcd

def downsample(inputpcd, voxel_size, voxel_size_2, n):
    downpcd = inputpcd.voxel_down_sample(voxel_size=voxel_size)
    pcd = np.asarray(downpcd.points).astype(np.float64)
    
    if pcd.shape[0] > n:
        """Drop or duplicate points so that pcd has exactly n points"""
        idx = np.random.permutation(pcd.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
        return pcd[idx[:n]].astype(np.float64)
    else:
        downpcd = inputpcd.voxel_down_sample(voxel_size=voxel_size_2)
        pcd = np.asarray(downpcd.points).astype(np.float64)
        if pcd.shape[0] > n:
            """Drop or duplicate points so that pcd has exactly n points"""
            idx = np.random.permutation(pcd.shape[0])
            if idx.shape[0] < n:
                idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
            return pcd[idx[:n]].astype(np.float64)

if __name__ == "__main__":
    train_file = sys.argv[1]
    train_seq = (np.loadtxt(train_file)).astype(np.int16)
    
    input_folder = sys.argv[2]
    assert(osp.exists(input_folder))
    save_folder = sys.argv[3]
    radius = int(sys.argv[4])
    pose_folder = sys.argv[5]
    partial_folder = sys.argv[6]

    if sys.argv[7] == 'save':
        if not osp.exists(save_folder):
            os.mkdir(save_folder)

    txt_file = sys.argv[8]
    txt = open(txt_file,"w")
    num = 0
    for idx in np.unique(train_seq[:,0]): # loop for different drive folders
        drive_name = "2013_05_28_drive_%04d_sync" %idx
        fused_cloud_folder = osp.join(input_folder, drive_name,"static")
        print("Opening the folder:" + fused_cloud_folder)
        pose_path = osp.join(pose_folder, drive_name, "poses.txt")
        assert(osp.exists(pose_path))
        poses = np.loadtxt(pose_path).astype(np.float)
        assert(osp.exists(fused_cloud_folder))
        seq_in_drive = train_seq[train_seq[:,0] == idx]
        seq_path = [osp.join(fused_cloud_folder,"%06d_%06d.ply" %(arr[1], arr[2])) \
                    for arr in seq_in_drive]

        seq_list = []
        seq_name = []
        for seq, idx in zip(seq_path, range(len(seq_path))):
            if not osp.exists(seq):
                pass
            try:
                input_pcd = o3d.io.read_point_cloud(seq) # read fused cloud
            except:
                print("Unable to load:" + seq)
                pass
            X_files = get_X_names(seq_in_drive[idx][1], seq_in_drive[idx][2]) # get start & end frame files
            save_path = osp.join(save_folder, "gt" ,drive_name + "_%06d_%06d_"%(seq_in_drive[idx][1], seq_in_drive[idx][2]))
            partial_path = osp.join(partial_folder, drive_name + "_%06d_%06d"%(seq_in_drive[idx][1], seq_in_drive[idx][2]))
            save_path_partial = osp.join(save_folder, "partial", drive_name + "_%06d_%06d_"%(seq_in_drive[idx][1], seq_in_drive[idx][2]))

            for x in X_files:
                if x in poses[:,0] and not x%2:
                    # downsample partial cloud
                    partial_file = osp.join(partial_path, str(x)+".dat")
                    if not osp.exists(partial_file):
                        pass  

                    # crop and downsample fused                  
                    pose_x = poses[poses[:,0]==x] # check the index of the pose
                    pose_matrix = pose_x[0,1:].reshape(3,4)
                    translation_vec = pose_matrix[:,3:].astype(np.float64)
                    cropped_pcd = crop_pointcloud(input_pcd, radius, translation_vec) # cropped pcd
                    downpcd = downsample(cropped_pcd, voxel_size_gt, voxel_size_gt_2, n_gt)

                    partial_pcd = points_to_pcd((np.loadtxt(partial_file)))
                    cropped_partial_pcd = crop_pointcloud(partial_pcd, radius, translation_vec) # CHANGES HERE 
                    downpcd_partial = downsample(cropped_partial_pcd, voxel_size_partial, voxel_size_partial_2, n_partial)

                    seq_list.append(1)
                    seq_name.append(save_path+str(x)+"\n")
                    # import pdb; pdb.set_trace()
                    try:
                        save = save_path + str(x)
                        print(downpcd.shape)
                        np.save(save, downpcd)

                        save_partial = save_path_partial + str(x)
                        print(downpcd_partial.shape)
                        np.save(save_partial, downpcd_partial)

                        print("saved at:"+ save)
                    except:
                        pass
        print("Number of files %04d"%len(seq_list))
        num+=len(seq_list)
        print("total number of files %04d"%num)
    print("Total number of training files: %04d"%num)
