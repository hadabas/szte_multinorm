#!/bin/bash

#SBATCH --job-name=SparsePGD_cifar100_2
#SBATCH --output=./logs/output_cifar100_seed2.log
#SBATCH --error=./logs/error_cifar100_seed2.log

source /shared/anaconda/anaconda/bin/activate
conda activate /shared/anaconda/jhadabas/.conda/envs/ramp


python ./adversarial_training/train.py --exp_name cifar100_seed2 --seed 2 --data_name cifar100 --data_dir ../../Adatbazis_mappak/cifar100 --model_name preactresnet --max_epoch 80 --batch_size 128 --lr 0.05 --train_loss adv --train_mode rand --patience 10 -k 60 --n_iters 20 --gpu 0