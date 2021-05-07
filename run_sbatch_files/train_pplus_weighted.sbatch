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

#SBATCH --time=23:00:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"



#SBATCH --error=job_files/overfit-8.%J.err
# write the error output to trainencoder.*jobID*.err

#SBATCH --output=job_files/overfit-8.%J.out
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

singularity exec /home/bharadwaj/envi/torch16-cuda10.simg python train.py --dataset kitti360 --dataset_path ../../dataset/scripts/4096-8192-kitti360/ --model final_training/weightedCD-full-data-pointnetPlus++2021-05-06T11:32:26.055105/network_80.pth --nepoch 200 --save_net 10 --num_points 8192 --num_point_partial 4096 --batchSize 8 --cuda True --save_folder_name pointnetPlusWeightedCD --pointnetPlus True --weightedCD True
echo DONE!