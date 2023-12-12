#!/bin/bash

#SBATCH -o results/LSTM_single_layer/train_and_test_restart-%j.out
#SBATCH -e results/LSTM_single_layer/train_and_test_restart-%j.err
#SBATCH --time 2880
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1


module load anaconda3/2023.07
module load cuda/10.2
module load openmpi/3.0.0

source activate py39

python3 -u /home/jshin029/CS229B/train_and_test.py /home/jshin029/CS229B /home/jshin029/CS229B/results/LSTM_single_layer /home/jshin029/CS229B/results/LSTM_single_layer/2776_model.pt 2776
