#!/bin/bash

#SBATCH --job-name=eval
#SBATCH --output=output.log
#SBATCH --error=error.log

source /shared/anaconda/anaconda/bin/activate
conda activate /shared/anaconda/jhadabas/.conda/envs/ramp


python eval-aa.py --data-dir ../../Adatbazis_mappak \
    --log-dir ./log \
    --desc hat-preactresnet18-cifar10-seed2 \
    --data cifar10 \
    --version custom