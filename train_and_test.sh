#!/bin/bash

#SBATCH -o logging/train_and_test-%j.out
#SBATCH -e logging/train_and_test-%j.err
#SBATCH --time 900



module load python/3.11.4

source /home/jshin029/CS229B/CPU_torch_3114/bin/activate

python3 /home/jshin029/CS229B/train_and_test.py /home/jshin029/CS229B/ /home/jshin029/CS229B/LSTM_enc_dec_output
