#!/bin/bash

#SBATCH --job-name=hat_cifar10_2
#SBATCH --output=./logs/output_hat_cifar10_seed2.log
#SBATCH --error=./logs/error_hat_cifar10_seed2.log

source /shared/anaconda/anaconda/bin/activate
conda activate /shared/anaconda/jhadabas/.conda/envs/ramp


python train.py --data-dir ../../Adatbazis_mappak \
    --log-dir ./log \
    --desc hat-preactresnet18-cifar10-seed2 \
    --data cifar10 \
    --model preact-resnet18 \
    --num-adv-epochs 80 \
    --helper-model std-preactresnet18-cifar10-seed2 \
    --beta 2.5 \
    --gamma 0.5 \
    --seed 2