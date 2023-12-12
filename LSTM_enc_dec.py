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
## jit edit
import torch.jit as jit

def maskedMSE(Y_pred,Y_test,mask,fillval):
    '''
    Calculates MSE based on masked data
    Inputs:
        Y_pred:   tensor of shape (seq_len,batch_size)
        Y_test:   tensor of shape (seq_len,batch_size)
        mask:     tensor of shape (seq_len,batch_size)
        fillval:  tensor of value to fill the masked array
    Outputs:
        MSE:      float of the average MSE overall batches
    '''

    masked_Y_pred = Y_pred.where(mask,fillval)
    masked_Y_test = Y_test.where(mask,fillval)

    batch_MSEs = torch.sum((masked_Y_pred - masked_Y_test)**2,dim=0)
    batch_MSEs = batch_MSEs / torch.sum(mask,dim=0)
    MSE = torch.mean(batch_MSEs)

    return MSE

## jit edit
# @jit.script
def maskedCrossEntropy(Y_pred,Y_test,mask,fillval):
    '''
    Calculates MSE based on masked data
    Inputs:
        Y_pred:   tensor of shape (seq_len,batch_size)
        Y_test:   tensor of shape (seq_len,batch_size)
        mask:     tensor of shape (seq_len,batch_size)
        fillval:  tensor of value to fill the masked array
    Outputs:
        MSE:      float of the average MSE overall batches
    '''

    batch_CEs = Y_test * torch.nan_to_num(torch.log(Y_pred+1e-9),nan=0.) + (1-Y_test)*torch.nan_to_num(torch.log(1-Y_pred+1e-9),nan=0.)
    batch_CEs = batch_CEs.where(mask,fillval)

    batch_CEs = -torch.sum(batch_CEs,dim=0)
    batch_CEs = batch_CEs / torch.sum(mask,dim=0)

    CE = torch.mean(batch_CEs)

    return CE

## jit edit
# @jit.script
def minimize_padding(X, Y, t_dim:int):
    '''
    remove extra zeros at the end of a tensor on the t_dim axis

    this is not implemented yet!!
    '''
    max_nonzero = X.nonzero()[:,t_dim].max()

    X = X[:max_nonzero,:,:]
    Y = Y[:max_nonzero,:,:]

    return X,Y

## jit edit
class encoder(torch.nn.Module):
# class encoder(jit.ScriptModule):

    def __init__(self, input_size, hidden_size, num_layers = 1, batch_first=False):
        '''
        Initializes the encoder LSTM
        Inputs:
            input_size:     length of each timepoint of the input X
            hidden_size:    length of the hidden states h[k], k = {0,...,num_layers-1}
            num_layers:     number of stacked LSTMs
        '''
        super(encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(input_size = input_size,
                                  hidden_size = hidden_size,
                                  num_layers = num_layers,
                                  batch_first = batch_first)

    # @jit.script_method
    def forward(self, x_input):
        '''
        Forward pass for the LSTM
        Inputs:
            x_input:    input of shape (sequence_length, batch_size, input_size)
        Outputs:
            lstm_out:   tensor of shape (sequence_length,batch_size,hidden_size) representing
                                output from last layer of LSTM for each t
            final:      tuple (h_n, c_n):
                            h_n is a tensor of shape (num_layers, hidden_size) for the hidden state at each layer
                            c_n is a tensor of shape (num_layers, hidden_size) for the cell state at each layer
        '''

        lstm_out, final = self.lstm(x_input)

        return lstm_out, final


## jit edit
class decoder(torch.nn.Module):
# class decoder(jit.ScriptModule):
    ''' Decodes hidden state output by encoder '''

    def __init__(self, input_size, output_size, hidden_size, num_layers = 1, batch_first=False):
        '''
        Initializes the decoder LSTM
        Inputs:
            input_size:     length of each timepoint of the input X
            output_size:    length of each timepoint of the output
            hidden_size:    length of the hidden states h[k], k = {0,...,num_layers-1}
            num_layers:     number of stacked LSTMs
        '''
        super(decoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(input_size = input_size,
                                  hidden_size = hidden_size,
                                  num_layers = num_layers,
                                  batch_first = batch_first)

        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.clip = torch.nn.Hardtanh(min_val=0., max_val=1.0)

    # @jit.script_method
    def forward(self, x_input,
                encoder_hidden_states):
        '''
        Forward pass for the LSTM
        Inputs:
            x_input:    input of shape (sequence_length, batch_size, input_size)
        Outputs:
            lstm_out:   tensor of shape (sequence_length,batch_size,hidden_size) representing
                                output from last layer of LSTM for each t
            final:      tuple (h_n, c_n):
                            h_n is a tensor of shape (num_layers, hidden_size) for the hidden state at each layer
                            c_n is a tensor of shape (num_layers, hidden_size) for the cell state at each layer
        '''

        lstm_out, final = self.lstm(x_input, encoder_hidden_states)

        output = self.linear(lstm_out)

        # when using cross entropy loss, output must be nonnegative
        output = self.clip(output)


        return output, final


## jit edit
class enc_dec(torch.nn.Module):
# class enc_dec(jit.ScriptModule):

    def __init__(self, input_size, output_size, hidden_size,
                enc_layers = 1, dec_layers = 1, sequence_length = 400,
                batch_size=512, batch_first=False):
        '''
        Initializes the encoder-decoder model
        Inputs:
            input_size:     length of each timepoint of the input X
            output_size:    length of each timepoint of the output
            hidden_size:    length of the hidden states h[k], k = {0,...,num_layers-1}
            enc_layers:     number of stacked LSTMs for encoder
            dec_layers:     number of stacked LSTMs for decoder
        '''

        super(enc_dec, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.batch_first = batch_first
        if self.batch_first:
            self.t_dim = 1
            self.batch_dim = 0
        else:
            self.t_dim = 0
            self.batch_dim = 1

        self.encoder = encoder(input_size = input_size, hidden_size = hidden_size,
                               num_layers = enc_layers, batch_first = batch_first)
        self.decoder = decoder(input_size = input_size+output_size, output_size = output_size,
                               hidden_size = hidden_size,
                               num_layers = dec_layers, batch_first = batch_first)

        self.output_zeros = torch.nn.Parameter(
                                torch.zeros(sequence_length, batch_size, output_size),requires_grad=False)

        self.zero_tensor = torch.nn.Parameter(
                                torch.tensor([0.]),requires_grad=False)



    def train_model(self, input_tensor, target_tensor,
                    test_input_tensor, test_target_tensor,
                    n_epochs, teacher_forcing_ratio = 0.5,
                    learning_rate = 0.01,
                    dynamic_tf = False, print_multiple=100, save_path='.',
                    restart_epoch=0):
        '''
        Trains the encoder-decoder model
        Inputs:
            input_tensor:              tensor of shape (sequence_length, batch_size, input_size) for the input data
            target_tensor:             tensor of shape (sequence_length, batch_size, output_size) for output data
            test_input_tensor:         tensor of shape (sequence_length, batch_size, input_size) for the test input data
            test_target_tensor:        tensor of shape (sequence_length, batch_size, output_size) for test output data
            n_epochs:                  number of epochs
            batch_size:                number of samples per gradient update (batch)
            teacher_forcing_ratio:     teacher forcing is applied with probability teacher_forcing_ratio;
                                          tfr = 0 ==> always recursive; tfr = 1 ==> always enforced
            learning_rate:             float >= 0 of the learning rate
            dynamic_tf:                float; determines the rate of teacher forcing decrease (0 implies no decrease)
            print_multiple: number of epochs after which to print the results
            save_path:      str designating filepath to which outputs are saved
            restart_epoch:  int designating the epoch at which training is restarted (after timeout)
        Outputs:
             losses:                   array of loss function for each epoch
        '''

        self.train()

        # initialize array of losses
        losses = np.full(n_epochs, np.nan)

        test_input_tensor,test_target_tensor = minimize_padding(
                                        test_input_tensor,
                                        test_target_tensor,
                                        t_dim = self.t_dim
                                        )

        optimizer = torch.optim.AdamW(self.parameters(), lr = learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate , steps_per_epoch=self.batch_size, epochs=n_epochs)


        # calculate number of batch iterations
        n_batches = int(input_tensor.shape[1] / self.batch_size)
        output_size = target_tensor.shape[2]

        test_losses = []

        start = time.time()

        for n_epoch in range(n_epochs):

            # dynamic teacher forcing
            if dynamic_tf and teacher_forcing_ratio > 0:
                tfr = teacher_forcing_ratio * np.sqrt(max((0,1-n_epoch/(n_epochs*0.8))))

            batch_loss = 0

            # randomizes the batch indices
            batch_idxs = torch.randperm(input_tensor.shape[1])

            for b in range(n_batches):
                # print(f"Training batch {b}/{n_batches} for epoch {n_epoch}/{n_epochs}; time elapsed: {time.time() - start :.2f} s")

                input_batch,target_batch = minimize_padding(
                                        input_tensor[:,batch_idxs[b:b+self.batch_size],:],
                                        target_tensor[:,batch_idxs[b:b+self.batch_size],:],
                                        t_dim = self.t_dim
                                        )

                # input_batch = input_tensor[:,b:b+self.batch_size,:]
                # target_batch = target_tensor[:,b:b+self.batch_size,:]

                output_list = []

                # zero the gradient
                optimizer.zero_grad(set_to_none=True)

                # encoder outputs
                encoder_output, encoder_hidden = self.encoder(input_batch)

                # decoder with teacher forcing

                unraveled_inputs = torch.unbind(input_batch,dim=self.t_dim)
                unraveled_targets = torch.unbind(target_batch,dim=self.t_dim)

                decoder_hidden = encoder_hidden

                # initialize first input for decoder
                decoder_output = torch.zeros(
                                    self.batch_size,
                                    output_size,
                                    device=input_tensor.device)

                # predict using mixed teacher forcing
                for t,(unraveled_input,unraveled_target) in enumerate(zip(unraveled_inputs,unraveled_targets)):

                    # predict with teacher forcing

                    if t == 0:
                        # don't teacher force first step - use zeros instead
                        decoder_input = torch.cat([decoder_output,
                                               unraveled_input],
                                               dim=self.batch_dim).unsqueeze(dim=self.t_dim)
                    elif np.random.random() < tfr:
                        decoder_input = torch.cat([unraveled_target,
                                               unraveled_input],
                                               dim=self.batch_dim).unsqueeze(dim=self.t_dim)
                    # predict recursively
                    else:
                        decoder_input = torch.cat([decoder_output,
                                                   unraveled_input],
                                                   dim=self.batch_dim).unsqueeze(dim=self.t_dim)

                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                    decoder_output = decoder_output.squeeze(self.t_dim)

                    output_list.append(decoder_output)

                outputs = torch.stack(output_list)

                # compute the loss
                with torch.cuda.amp.autocast():
                    loss = maskedCrossEntropy(outputs.squeeze(-1),
                                     target_batch.squeeze(-1),
                                     input_batch.any(axis=2),
                                     self.zero_tensor)

                batch_loss += loss.item()

                # backpropagation
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), 5)

                optimizer.step()
                scheduler.step()

            # loss for epoch
            batch_loss = batch_loss/n_batches
            losses[n_epoch] = batch_loss


            if n_epoch % print_multiple == 0:
                if n_epoch % (print_multiple*5) == 0:
                    # save model for continued training in case training reaches time limit
                    torch.save(self.state_dict(),f"{save_path}/{n_epoch+restart_epoch}_model.pt")

                self.eval()
                with torch.no_grad():
                    pred_Y = self.predict(test_input_tensor,
                                          output_size)

                    test_loss = maskedCrossEntropy(pred_Y.squeeze(-1),
                                         test_target_tensor.squeeze(-1),
                                         test_input_tensor.any(axis=2),
                                         self.zero_tensor)


                test_losses.append(test_loss.item())

                self.train()

                print("Epoch ",n_epoch+restart_epoch,"Train Batch Loss: ",np.round(batch_loss,3),"Test Loss: ", np.round(test_loss.item(),3),"; Time elapsed: ",np.round(time.time() - start,2), flush=True)
            else:
                print("Epoch ",n_epoch+restart_epoch,"Train Batch Loss: ",np.round(batch_loss,3),"; Time elapsed: ",np.round(time.time() - start,2), flush=True)



        return losses,test_losses

    def predict(self, input_tensor, output_size):
        '''
        Makes a prediction!
        Inputs:
            input_tensor:      tensor of shape (sequence_length, batch_size, input_size) with the input data
            output_size:       dimensions of the output
        Outputs:
            output_tensor:     tensor of shape (sequence_length, batch_size, output_size) containing predicted
                                  values (predicted recursively)
        '''

        with torch.no_grad():
            self.eval()
            outputs = torch.zeros(*input_tensor.shape[:2], output_size).to(input_tensor.device)

            encoder_output, encoder_hidden = self.encoder(input_tensor)

            unraveled_inputs = torch.unbind(input_tensor,dim=0)
            decoder_hidden = encoder_hidden

            # initialize first input for decoder
            decoder_output = torch.zeros(input_tensor.shape[1],output_size).to(input_tensor.device)

            for t,unraveled_input in enumerate(unraveled_inputs):
                decoder_output = decoder_output.squeeze(self.t_dim)
                decoder_input = torch.cat([decoder_output,unraveled_input], dim=1).unsqueeze(self.t_dim)
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[t,:,:] = decoder_output

            output_tensor = outputs.detach()

        return output_tensor

if __name__ == "__main__":

    pass
