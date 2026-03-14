#!/bin/bash

#SBATCH --job-name=Eval_2
#SBATCH --output=./eredmenyek/cifar10/union_avg_preact18_cifar10_seed2_output.log
#SBATCH --error=./eredmenyek/cifar10/union_avg_preact18_cifar10_seed2_error.log

source /shared/anaconda/anaconda/bin/activate
conda activate /shared/anaconda/jhadabas/.conda/envs/ramp


python ./eval_all_UNION.py --model_name ../Vanilla_halok/PreActResNet18/Cifar10/Union/AVG_cifar10_seed2/iter_80.pt \
                    --data_dir ../Adatbazis_mappak/cifar10 \
                    --dataset cifar10 \
                    --run_border \
                    --run_border_inner