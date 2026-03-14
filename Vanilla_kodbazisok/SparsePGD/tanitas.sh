#!/bin/bash

#SBATCH --job-name=SparsePGD_cifar10_2
#SBATCH --output=./logs/output_cifar10_seed2.log
#SBATCH --error=./logs/error_cifar10_seed2.log

source /shared/anaconda/anaconda/bin/activate
conda activate /shared/anaconda/jhadabas/.conda/envs/ramp


python ./adversarial_training/train.py --exp_name cifar10_seed2 --seed 2 --data_name cifar10 --data_dir ../../Adatbazis_mappak/cifar10 --model_name preactresnet --max_epoch 80 --batch_size 128 --lr 0.05 --train_loss adv --train_mode rand --patience 10 -k 120 --n_iters 20 --gpu 0