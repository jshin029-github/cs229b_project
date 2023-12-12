#!/bin/bash

#SBATCH -o logging/generate_sequences-%j.out
#SBATCH -e logging/generate_sequences-%j.err
#SBATCH --time 1000
#SBATCH --ntasks 16
#SBATCH --cpus-per-task 1



module load python/3.11.4
source /home/jshin029/CS229B/CPU_torch_3114/bin/activate

python3 /home/jshin029/CS229B/generate_sequences.py /home/jshin029/CS229B/gen_seqs
