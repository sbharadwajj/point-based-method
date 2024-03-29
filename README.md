## Point-Based Methods

This repository contains a Point-Based Method that is evaluated on KITTI-360

#### 1) Envrionment & prerequisites

Please check `torch-16-cuda10.recipe` for singuarity recipe to build the environment used for the baseline

- Pytorch 1.6.0
- CUDA 10.1
- Python 3.7
- chamfer_distance



File descriptions

- chamfer_torch.py: chamfer loss defined by torch3d
- data_utils.py - contains data augmentation from completion3d page
- pointnet2_utils - has the `PointSetAbstraction` required for the encoder of PointNet++
- val_accu_comp - is a validation file that saves pointwise loss to calculate accuracy and completeness later on

#### 2) Compile

Create singularity image from the recipe 

#### 3) Train or validate

To train:

1. For PointNet++ run `run_sbatch_files/train_pplus_normal.sbatch` and change these accordingly:
```
singularity exec [path-to-simg]/torch16-cuda10.simg python [path-to-train.py]train.py --dataset kitti360 --dataset_path [path-to-dataset]/4096-8192-kitti360/ --model [path-to-load-previous-weights]/network_80.pth --nepoch 200 --save_net 10 --num_points 8192 --num_point_partial 4096 --batchSize 8 --save_folder_name pointnetPlus-exp1 --cuda True --pointnetPlus True
```

2. For PointNet++ with weighted CD run `run_sbatch_files/train_pplus_weighted.sbatch` with an additional `--weightedCD True`

3. For PointNet run `run_sbatch_files/train_normal.sbatch`:
```
singularity exec [path-to-simg]/torch16-cuda10.simg python [path-to-train.py]train.py --dataset kitti360 --dataset_path [path-to-dataset]/4096-8192-kitti360/ --model [path-to-load-previous-weights]/network_80.pth --nepoch 200 --save_net 10 --num_points 8192 --num_point_partial 4096 --save_folder_name pointnet-exp1 --batchSize 8 --cuda True
```

4. For PointNet with weighted CD run the above with an additional `--weightedCD True`


Note:
- A folder is created in the default path called `final_training` and the network weights are saved in them
- Master branch contains the final code used for all the experiments

