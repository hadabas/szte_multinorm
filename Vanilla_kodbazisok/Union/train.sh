#!/bin/bash

#SBATCH --job-name=Union_AVG_2
#SBATCH --output=./logs/output_union_avg_cifar100_2.log
#SBATCH --error=./logs/error_union_avg_cifar100_2.log

source /shared/anaconda/anaconda/bin/activate
conda activate /shared/anaconda/jhadabas/.conda/envs/ramp


python ./union/CIFAR100_custom/train.py -gpu_id 0 -model 4 -batch_size 128