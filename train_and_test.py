#!/usr/bin/env python3
"""
Contains the classes necessary to train an encoder-decoder with an LSTM-LSTM model
"""

__author__ = "John Shin"
__version__ = "0.1.0"
__license__ = "MIT"


import numpy as np
import torch
import time
import sys
from LSTM_enc_dec import *

if __name__ == "__main__":

    args = sys.argv[1:]

    if (len(args) != 2) & (len(args) != 4):
        print('Usage: train_and_test <path_to_data> <path_to_output> <optional: path_to_intermediate_model> <optional: model number>')
        print(args)
        exit()


    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Selected device: {device}", flush=True)

    print('Importing data', flush=True)
    # Import data
    all_X = torch.load(f"{args[0]}/all_X.pt")
    all_Y = torch.load(f"{args[0]}/all_Y.pt")

    # Split data into a multiple of 1024 for hardware(?) reasons
    train_X = all_X[:,:1024*97,:].to(device)
    train_Y = all_Y[:,:1024*97,:].to(device)

    test_X = all_X[:,1024*97:,:].to(device)
    test_Y = all_Y[:,1024*97:,:].to(device)

    print('Building Model', flush=True)
    model = enc_dec(input_size = train_X.shape[2],
                    output_size = 1,
                    hidden_size = 64,
                    enc_layers = 1,
                    dec_layers = 1,
                    sequence_length = train_X.shape[0],
                    batch_size = 512)

    if len(args) == 4:
        print('Loading Model',flush=True)
        model.load_state_dict(torch.load(f"{args[2]}"))
        restart_epoch = int(args[3])+1

    model = model.to(device)

    print(f"Training model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters", flush=True)

    loss = model.train_model(train_X, train_Y,
                             test_X, test_Y,
                             n_epochs = 2000,
                             teacher_forcing_ratio = 0.1,
                             learning_rate = 0.01, dynamic_tf = 0.001,
                             print_multiple = 5, save_path = args[1],
                             restart_epoch = restart_epoch)

    torch.save(model.state_dict(),f"{args[1]}/final_state.pt")

    print(loss)

    pred_Y = model.predict(test_X,test_Y.shape[-1])
    print('test loss is ', maskedCrossEntropy(pred_Y.squeeze(-1),
                                     test_Y.squeeze(-1),
                                     test_X.any(axis=2),
                                     model.zero_tensor), flush=True)

    torch.save(pred_Y,f"{args[1]}/predicted_Y.pt")
