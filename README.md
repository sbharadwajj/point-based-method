## Baseline for pointcloud extrapolation


### Changes made:
- added lr scheduler
- removed feature_transform=True


#### 1) Envrionment & prerequisites

- Pytorch 1.6.0
- CUDA 10.1
- Python 3.7
- chamfer_distance

#### 2) Compile

Create singularity image from the recipe 

#### 3) Train or validate

change path in `dataset.py`

check sbatch file for training instructions.

Run `python3 val.py` to validate the model or `python3 train.py` to train the model from scratch.
