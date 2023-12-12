#!/usr/bin/env python3
"""
Contains the classes necessary to train an encoder-decoder with an LSTM-LSTM model
"""

__author__ = "John Shin"
__version__ = "0.1.0"
__license__ = "MIT"


import sys
import RNA
import numpy as np
import torch
from joblib import Parallel, delayed

def makeHelix(max_len,
                avg_helix_len = 3,
                avg_loop_len = 2,
                paired_emissions = [.80,.18,.02],
                ):


    rng = np.random.uniform()

    loop_len = np.random.poisson(lam=avg_loop_len)+2
    max_len -= loop_len

    if max_len < 6:
        return ''

    helix_len = np.random.poisson(lam=avg_helix_len)+3
    helix_counter = 0

    left_seq = ''
    right_seq = ''

    while (helix_counter < helix_len) & (len(left_seq) + len(right_seq) < max_len - 1):

        if np.random.uniform() < paired_emissions[0]:
            left_seq += '('
            right_seq = ')' + right_seq
        elif rng < sum(paired_emissions[:2]):

            max_new_seq = max_len - len(left_seq) - len(right_seq)

            if max_new_seq > 8:
                if np.random.uniform() < 0.5:
                    left_seq += makeHelix(max_new_seq)
                else:
                    right_seq = makeHelix(max_new_seq) + right_seq

        elif np.random.uniform() < 0.5:
            left_seq += '.'
        else:
            right_seq = '.' + right_seq

        helix_counter += 1

    return left_seq + '.'*loop_len + right_seq

def CFRNASSM(seq_len,
             state_space = {0:'paired',1:'unpaired'},
             avg_helix_len = 3,
             avg_loop_len = 3,
             state_transitions = np.array([[.5,.5],[.3,.7]]),
             paired_emissions = [.95,.05]):

    current_state = int(np.random.uniform() > 0.5)

    left_seq = ''
    right_seq = ''

    while len(left_seq) + len(right_seq) < seq_len - 1:

        rng = np.random.uniform()

        helix_len = np.random.poisson(lam=avg_helix_len)+3
        loop_len = np.random.poisson(lam=avg_loop_len)

        if current_state == 0:
            if rng < 0.5:
                left_seq += makeHelix(seq_len - len(left_seq) - len(right_seq))
            else:
                right_seq = makeHelix(seq_len - len(left_seq) - len(right_seq)) + right_seq
        else:
            loop_counter = 0

            while (loop_counter < loop_len)\
                    & (len(left_seq) + len(right_seq) < seq_len - 1):

                if rng < 0.5:
                    left_seq += '.'
                else:
                    right_seq = '.' + right_seq

                loop_counter += 1

        current_state = int(state_transitions[current_state,0] < rng)

    seq = left_seq + right_seq

    if len(seq) < seq_len:
        seq += '.'

    return seq

def parseDotBracket(dot_bracket,
                    bp_dict = {'A':'U','U':'A','G':'C','C':'G'},
                    GU_prob = 0.1,
                    GU_dict = {'A':'U','U':'G','G':'U','C':'G'}
                   ):
    seq = [None]*len(dot_bracket)

    helix_stack = []

    max_iter = len(dot_bracket)
    counter = 0

    while (len(seq) > 0) & (counter < max_iter):
        nuc = np.random.choice(['A','U','G','C'])

        if dot_bracket[0] == '.':
            seq[counter] = nuc
            dot_bracket = dot_bracket[1:]

        elif dot_bracket[0] == '(':
            helix_stack += [(counter,np.random.choice(['A','U','G','C','G','C']))]
            dot_bracket = dot_bracket[1:]

        elif dot_bracket[0] == ')':
            seq[helix_stack[-1][0]] = helix_stack[-1][1]

            if np.random.uniform() < GU_prob:
                seq[counter] = GU_dict[helix_stack[-1][1]]
            else:
                seq[counter] = bp_dict[helix_stack[-1][1]]

            helix_stack = helix_stack[:-1]
            dot_bracket = dot_bracket[1:]

#         print(dot_bracket,[f[0] for f in helix_stack])
        counter += 1

    return ''.join(seq)

def makeSamplesAndWrite(output_path,i,n_seqs = 1000,write_outputs=False):

    output = ''

    seq_lens = np.random.randint(150,401,size=n_seqs)

    for seq_len in seq_lens:

        struct = CFRNASSM(seq_len)
        seq = parseDotBracket(struct)
        fc = RNA.fold_compound(seq)
        fc.pf()
        bpp = np.sum(np.array(fc.bpp()) + np.array(fc.bpp()).T,axis=1)[1:]

        output += f"{seq}\t{','.join(bpp.astype('str'))}\n"

    output = output.rstrip()

    if write_outputs:
        with open(f'{output_path}/{i}.tsv','w') as f:
            f.write(output)

    return output

def base_dict(s):
    one_hot_encode = torch.zeros(1,4)
    if s == 'A':
        one_hot_encode[:,0] = 1
    elif s == 'U':
        one_hot_encode[:,1] = 1
    elif s == 'G':
        one_hot_encode[:,2] = 1
    elif s == 'C':
        one_hot_encode[:,3] = 1

    return one_hot_encode

if __name__ == "__main__":

    args = sys.argv[1:]

    if len(args) != 1:
        print('Usage: train_and_test <path_to_output>')
        exit()

    output_path = args[0]

    outputs = Parallel(n_jobs=-1,verbose=10)(delayed(makeSamplesAndWrite)(output_path,i) for i in range(100))

    outputs = '\n'.join(outputs)
    outputs = outputs.split('\n')

    all_X = torch.zeros(400,len(outputs),4)
    all_Y = torch.zeros(400,len(outputs),1)

    for i,output in enumerate(outputs):
        seq,bpp = output.split('\t')
        all_X[:len(seq),i,:] = torch.cat([base_dict(s) for s in seq],dim=0)
        all_Y[:len(seq),i,:] = torch.tensor(np.array(bpp.split(','),dtype=np.float32)).unsqueeze(1)

    torch.save(all_X,f"{output_path}/all_X.pt")
    torch.save(all_Y,f"{output_path}/all_Y.pt")
