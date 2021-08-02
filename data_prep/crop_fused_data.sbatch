#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=TutorialJob

#resources:

#SBATCH --cpus-per-task=3
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=day
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=4G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=10:00:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=createdata.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=createdata.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=***@uni-tuebingen.de
# your mail address

####
#b) copy all needed data to the jobs scratch folder
####

####
#c) Execute your tensorflow code in a specific singularity container
#d) Write your checkpoints to your home directory, so that you still have them if your job fails
#cnn_minst.py <model save path> <mnist data path>
####

# singularity exec /home/bharadwaj/envi/py3-6-open3d-ubuntu16.simg python3 create_train_pair.py 2013_05_28_drive_train_only.txt <fused-cloud-folder-kitti360> <save-folder> radius <path-to-poses-folder> <path-to-sparse-cloud-folder> save train_list.txt
singularity exec /home/bharadwaj/envi/py3-6-open3d-ubuntu16.simg python3 create_train_pair.py 2013_05_28_drive_train_only.txt ../data_3d_semantics/ 4096-8192-30-kitti360/train 30 ../KITTI-360/data_poses/ ../final_training/train_partial/ save train_list.txt

echo DONE!
