#!/bin/bash

#SBATCH --job-name=Cifar100_2
#SBATCH --output=./logs/output_cifar100_seed2.log
#SBATCH --error=./logs/error_cifar100_seed2.log

source /shared/anaconda/anaconda/bin/activate
conda activate /shared/anaconda/jhadabas/.conda/envs/ramp


python RAMP_cifar100.py --lr-max 0.05  --lr-schedule=static --at_iter 10 --epochs 80 --save_freq 10 --eval_freq 10 --fname RAMP_cifar100_seed_2 --kl --max --gp --lbd 5 --seed 2 --final_eval