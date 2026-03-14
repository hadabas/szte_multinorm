#!/bin/bash

#SBATCH --job-name=Eval_1
#SBATCH --output=./eredmenyek/cifar100/spgd_preact18_cifar100_chkpt_trades_output.log
#SBATCH --error=./eredmenyek/cifar100/spgd_preact18_cifar100_chkpt_trades_error.log

source /shared/anaconda/anaconda/bin/activate
conda activate /shared/anaconda/jhadabas/.conda/envs/ramp


python ./eval_all_SPGD.py --model_name ../Vanilla_halok/PreActResNet18/Cifar100/SparsePGD/chkpt_cifar100_trades/cifar100_trades_k60_rand_20iters_100epochs_10patience.pth \
                    --data_dir ../Adatbazis_mappak/cifar100 \
                    --dataset cifar100 \
                    --run_border \
                    --run_border_inner