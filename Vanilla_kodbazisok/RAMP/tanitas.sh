#!/bin/bash

#SBATCH --job-name=Cifar10_0
#SBATCH --output=./logs/output_cifar10_seed0.log
#SBATCH --error=./logs/error_cifar10_seed0.log

source /shared/anaconda/anaconda/bin/activate
conda activate /shared/anaconda/jhadabas/.conda/envs/ramp


python RAMP.py --lr-max 0.05  --lr-schedule=static --at_iter 10 --epochs 80 --save_freq 10 --eval_freq 10 --fname RAMP_cifar10_seed_0 --kl --max --gp --lbd 5 --seed 0