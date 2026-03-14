#!/bin/bash

#SBATCH --job-name=hat_cifar10_2
#SBATCH --output=./logs/output_cifar10_seed2.log
#SBATCH --error=./logs/error_cifar10_seed2.log

source /shared/anaconda/anaconda/bin/activate
conda activate /shared/anaconda/jhadabas/.conda/envs/ramp


python train.py --data-dir ../../Adatbazis_mappak \
    --log-dir ./log \
    --desc std-preactresnet18-cifar10-seed2 \
    --data cifar10 \
    --model preact-resnet18 \
    --num-std-epochs 80 \
    --seed 2