#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=TutorialJob

#resources:

#SBATCH --cpus-per-task=4
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=test
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=12G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=00:15:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"



#SBATCH --error=job_files/valca.%J.err
# write the error output to trainencoder.*jobID*.err

#SBATCH --output=job_files/valca.%J.out
# write the standard output to trainencoder.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=***@uni-tuebingen.de
# your mail address

####
#b) copy all needed data to the jobs scratch folder
####

# cp -R /home/bharadwaj/dataset/final_training/ /scratch/$SLURM_JOB_ID/

####
#c) Execute your tensorflow code in a specific singularity container
#d) Write your checkpoints to your home directory, so that you still have them if your job fails
#cnn_minst.py <model save path> <mnist data path>
####

singularity exec /home/bharadwaj/envi/torch16-cuda10.simg python val_accu_comp.py --dataset_path ../../dataset/scripts/4096-8192-kitti360/ --model overfit_kitti360/pointnet++2021-05-04T21\:23\:21.163209/network_5950.pth --num_points 8192 --cuda True --pointnet True
echo DONE!