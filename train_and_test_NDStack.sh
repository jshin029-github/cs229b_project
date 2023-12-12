#!/bin/bash

#SBATCH -o results/NDStack/train_and_test-%j.out
#SBATCH -e results/NDStack/train_and_test-%j.err
#SBATCH --time 2880
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4

module load anaconda3/2023.07
module load cuda/10.2
module load openmpi/3.0.0

source activate py39

python3 -u /home/jshin029/CS229B/train_and_test_NDStack.py /home/jshin029/CS229B /home/jshin029/CS229B/results/NDStack
