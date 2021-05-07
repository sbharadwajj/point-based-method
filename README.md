## Baseline for pointcloud extrapolation


### Changes made:
- added pointnet plus plus encoder
- fixed data aug to z-axis
- added weightedCD for both completeness and accuracy
- (disabled LR scheduler for now)/ set is at a higher epoch
- lr 10e5 is slow, but converges
- weightedCD is now "mean" reduced for point reduction
- tried feature transform and its bad for our case


#### 1) Envrionment & prerequisites

- Pytorch 1.6.0
- CUDA 10.1
- Python 3.7
- chamfer_distance

File descriptions

- chamfer_torch.py - has chamfer loss defined by torch3d
- data_utils.py - contains data augmentation from completion3d page
- pointnet2_utils - has the `PointSetAbstraction` required for the encoder of PointNet++

#### 2) Compile

Create singularity image from the recipe 

#### 3) Train or validate

To train:

1. For PointNet++ run `run_sbatch_files/train_pplus_normal.sbatch` and change these accordingly:
```
singularity exec [path-to-simg]/torch16-cuda10.simg python [path-to-train.py]train.py --dataset kitti360 --dataset_path [path-to-dataset]/4096-8192-kitti360/ --model [path-to-load-previous-weights]/network_80.pth --nepoch 200 --save_net 10 --num_points 8192 --num_point_partial 4096 --batchSize 8 --cuda True --save_folder_name pointnetPlusNormalCD --pointnetPlus True
```

2. For PointNet++ with weighted CD run `run_sbatch_files/train_pplus_weighted.sbatch` and make similar changes as above

Note:
A folder is created in the default path called 'final_training' and the network weights are saved in them

