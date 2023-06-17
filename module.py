from __future__ import annotations
from lzma import is_check_supported
import math
from nis import cat
from operator import is_not
from re import S
from turtle import forward, position
from typing import Iterator, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor as T
import torch.nn.functional as F
import numpy as np
import random
from collections import deque 
import time
import os, shutil

tmp_fname = '.tmp.pt'

class TurboSequencer(nn.Module):
    """Module that can decode sequences and teacher force."""
    def __init__(self, sequence_encoder:nn.Module, dim_token:int, dim_static:int):
        """_summary_

        Args:
            sequence_encoder (nn.Module): Input shape of (seq_len, batch_size, dim_token+dim_static), Output shape of (batch_size, dim_token)
            dim_token (int): dimension of the tokens
            dim_static (int): dimension of static info (to condition the generation on)
        """
        super().__init__()
        #super().__init__(input_size=dim_token+dim_static, hidden_size=dim_hidden, **kwargs)
        #self.proj_layer = nn.Linear(dim_hidden, dim_token)
        self.dim_token=dim_token
        self.dim_static=dim_static
        self.sequence_encoder=sequence_encoder

    def forward(self, x):
        #out, (h_n, c_n) = super().forward(x)
        #out = out.mean(dim=0) # aggregate time dim. => (batch_size, dim_hidden)
        out = self.sequence_encoder(x)
        #out = self.proj_layer(out) # (batch_size, dim_token)
        return out

    def get_device(self):
        """ returns the device that the model is on (assuming all parameters are on the same device) """
        return next(self.parameters()).device

    def decode(self, x_static, teacher_force_y=None, decoding_steps=3, post_process=lambda x: x):
        """Returns a list of token predictions over the amount of decoding steps/length of teacher forcing sequence (optional).
        The static info is in the first position (:static_dim).

        Args:
            x_static (Tensor): Shape (batch_size, static_dim). Static information (does not change with time steps).
            teacher_force_y (Tensor, optional): Shape (seq_len, batch_size, dim_token). Defaults to None.
            decoding_steps (int, optional): Number of decoding steps (only works when teacher_force_y==None). Defaults to 3.
        """
        err_msg = "teacher forcing must be of shape (seq_len, batch_size, dim_token)"
        if teacher_force_y is not None: 
            assert teacher_force_y.dim() == 3, err_msg+": must provide 3 dimensions"
            assert x_static.shape[0] == teacher_force_y.shape[1], err_msg+": batch size must match"
            assert teacher_force_y.shape[2]==self.dim_token, err_msg+": incorrect token dimension"
            decoding_steps = teacher_force_y.shape[0]
        assert x_static.shape[1]==self.dim_static, err_msg+": incorrect static dimension"
        batch_size = x_static.shape[0]

        results_final = [] # to be used for loss computation
        results_intermediate = [] # to be used as pretext during decoding (e.g. reembedded token prediction)
        for step in range(decoding_steps):
            # set up the input tensor to the RNN x: 
            # - if teacher forcing: correct previous outputs
            # - else: previously predicted outputs
            x = torch.zeros(1+step, batch_size, self.dim_token+self.dim_static).to(self.get_device())
            for i in range(1+step):
                x[i, :, :self.dim_static] = x_static.clone()
            for i in range(step):
                if teacher_force_y is None:
                    x[1+i, :, self.dim_static:] = results_intermediate[i].clone()
                else:
                    x[1+i, :, self.dim_static:] = teacher_force_y[i, :, :].clone()
            out = self.forward(x)            
            results_final.append(out)
            results_intermediate.append(post_process(out.clone()))
        results_final = torch.stack(results_final)
        return results_final

class LSTMEncoder(nn.LSTM):
    def __init__(self, dim_input, dim_output, **kwargs):
        if 'hidden_size' not in kwargs:
            kwargs['hidden_size'] = 16
        super().__init__(input_size=dim_input, **kwargs)
        if 'bidirectional' in kwargs and kwargs['bidirectional']: 
            self.proj_layer = nn.Linear(kwargs['hidden_size']*2, dim_output)
        else:
            self.proj_layer = nn.Linear(kwargs['hidden_size'], dim_output)

    def forward(self, x):
        out, (h_n, c_n) = super().forward(x)
        out = out.mean(dim=0) # aggregate time dim. => (batch_size, dim_hidden*D)
        out = self.proj_layer(out) # (batch_size, dim_token)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100, batch_first:bool=False):
        super().__init__()
        #self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: T) -> T:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        if self.batch_first: 
            x = x.transpose(0, 1)
        pos = self.pe[:x.size(0)]
        x = x + pos
        #return self.dropout(x)
        if self.batch_first: 
            x = x.transpose(0, 1)
        return x

class TransformerModel(nn.Module):
    def __init__(self, dim_model:int, max_len:int=100, aggregate=False, hyperparams:dict={}, conv_start:bool=False):
        super().__init__()
        _hp = hyperparams.copy()

        num_layers = _hp.get('num_layers', 2)
        try: del _hp['num_layers']
        except KeyError: pass
        
        if conv_start: 
            self.conv_start = nn.Conv1d(in_channels=1, )
        else: 
            self.conv_start = None

        self.batch_first = _hp.get('batch_first', False)
        self.pos_enc = PositionalEncoding(dim_model, max_len, self.batch_first)
        self.aggregate = aggregate
        encoder_layer = nn.TransformerEncoderLayer(dim_model, **_hp)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, aggregate_override=None):
        """ Returns top level embedding as mean of the sequence """
        x = self.pos_enc(x)
        x = self.transformer(x)
        if aggregate_override is None:
            aggregate_override = self.aggregate
        if aggregate_override:
            x = torch.mean(x, dim=0)
        return x

#m = TransformerModel(dim_model=8, max_len=4, aggregate=False, hyperparams={'num_layers': 2, 'nhead': 4, 'dim_feedforward': 32}, conv=True)

class CharacterLevelCNN(nn.Module):
    # see https://github.com/uvipen/Character-level-cnn-pytorch
    def __init__(
        self, 
        dim_token=68,
        dim_out=14, 
        max_context_size=1014, 
        n_conv_filters=256,
        n_fc_neurons=1024,
        ):
        super(CharacterLevelCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(dim_token, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))
        self.conv2 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))
        self.conv3 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))

        dimension = int((max_context_size - 96) / 27 * n_conv_filters)
        self.fc1 = nn.Sequential(nn.Linear(dimension, n_fc_neurons), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(n_fc_neurons, n_fc_neurons), nn.Dropout(0.5))
        self.fc3 = nn.Linear(n_fc_neurons, dim_out)

        if n_conv_filters == 256 and n_fc_neurons == 1024:
            self._create_weights(mean=0.0, std=0.05)
        elif n_conv_filters == 1024 and n_fc_neurons == 2048:
            self._create_weights(mean=0.0, std=0.02)

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def forward(self, input):
        input = input.transpose(1, 2)
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)

        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output

#cnn = CharacterLevelCNN()
#print(cnn(torch.rand(8, 1014, 68)).shape)
#x = torch.rand(5, 10, 12)
#out = m(x)
#print(out.shape)
#exit()

class TransformerEncoder(TransformerModel):
    def __init__(self, dim_input, dim_output, **kwargs):
        super().__init__(dim_model=dim_input, **kwargs)
        self.proj_layer = nn.Linear(dim_input, dim_output)

    def forward(self, x):
        out = super().forward(x)
        out = self.proj_layer(out)
        return out

#m = TransformerEncoder(8, 4, num_layers=2, nhead=4, dim_feedforward=32)
#x = torch.rand(5, 10, 12)
#out = m(x)
#print(out.shape)
#exit()

class ReconstructionModel(nn.Module):
    def __init__(self):
        super().__init__()

    def count_parameters(self):
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def loss_fn(self, *args):
        raise Exception("implement me pls")

    def forward(self, x):
        raise Exception("implement me pls")

    def get_device(self):
        """ returns the device that the model is on (assuming all parameters are on the same device) """
        return next(self.parameters()).device

    def train_batch(self, batch, optim, args=[], kwargs={}) -> Union[T, None]:
        self.train()
        optim.zero_grad()
        res = self.forward(batch, *args, **kwargs)
        loss = self.loss_fn(*res)
        if loss is None: 
            return None
        retval = loss.item()
        loss.backward()
        optim.step()
        return retval

    def eval_batch(self, batch):
        self.eval()
        with torch.no_grad():
            res = self.forward(batch)
        loss = self.loss_fn(*res)
        retval = loss.item()
        return loss.item()
    
    def eval_batches(self, batches):
        losses = [self.eval_batch(batch) for batch in batches]
        return np.mean(losses).item(), losses
        
    def _try_record_best_loss(self, val, verbose, min_improvement):
        if self.best_loss > val + min_improvement: 
            if verbose: 
                print(f"new best loss ({self.best_loss:.6f} -> {val:.6f})")
            self.best_loss = val 
            self.epochs_since_new_best = 0
            if self.save_best is None: 
                torch.save(self.state_dict(), tmp_fname)
            else: 
                torch.save(self.state_dict(), self.save_best)
            return True
        self.epochs_since_new_best += 1
        return False

    def _record_train_loss(self, verbose):
        loss = np.mean(self.loss_accum).item()
        batch_time = np.mean(self.time_accum).item()
        self.losses_train.append(loss)
        self.loss_accum = []
        self.time_accum = []
        if verbose: 
            batch_info = '' if self.batch_i is None else f"\tBatch {self.batch_i:02d}"
            print(f"Epoch {self.epoch:02d}{batch_info}\t{batch_time:.4f}s/batch\ttrain_loss = {loss:.4f}")
        return loss

    def _record_test_loss(self, batches_test, verbose):
        loss = self.eval_batches(batches_test)[0]
        self.losses_test.append(loss)
        if verbose: 
            batch_info = '' if self.batch_i is None else f"\tBatch {self.batch_i:02d}"
            print(f"Epoch {self.epoch:02d}{batch_info}\ttest_loss = {loss:.4f}")
        return loss

    def train_batches(
        self, batches_train:Iterator, epochs:int, lr:float, 
        batches_test:Iterator=None, 
        test_every:int=None, 
        print_every:int=None, 
        verbose:bool=True, 
        optim:torch.optim.Optimizer=None, 
        patience:int=5, 
        min_improvement:float=.01, 
        save_best:str=None) -> Union[Iterator, Iterator, torch.optim.Optimizer]:
        """_summary_

        Args:
            batches_train (iterable): every item (batch) will be used as input to the forward method
            epochs (int): when > 0, trains for the amount of epochs; else trains until convergence
            lr (float): learning rate 
            batches_test (iterable, optional): Just like batches_train. Defaults to None.
            test_every (int, optional): Specifies the number of batches to train inbetween evaluations. If None, evaluate after every epoch. Defaults to None.
            print_every (int, optional): Just like test_every, but for printing training state. Defaults to None.
            verbose (bool, optional): Verbosity. Defaults to True.
            optim (torch.optim.Optimizer, optional): If not none, will use this as optimizer; otherwise, initializes an ADAM optimizer. Defaults to None.
            patience (int, optional): If there is no improvement over this amount of batches, training stops. Defaults to 5.
            min_improvement (float, optional): Improvement counts if it is greater than this value. Defaults to .01.
            save_best (str, optional): Path to save the best state of the model. Defaults to None.

        Returns:
            tuple: losses_train, losses_test, optim
        """
        
        if optim is None:
            optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.losses_train = [] # if print_every is None, entry is the average loss in the epoch, else the average loss in the last print_every batches
        self.losses_test = [] # if test_every is None, entry is the test loss after each epoch, else after last test_every batches
        self.batch_i, self.epoch = 0, -1
        self.loss_accum = [] # entry is loss per batch; reset every epoch if print_every is None, else after print_every batches
        self.time_accum = [] # entry is time per batch; reset every epoch if print_every is None, else after print_every batches
        self.best_loss = np.inf # lowest loss 
        self.save_best=save_best
        self.epochs_since_new_best = 0
        until_convergence = epochs is None or epochs < 1
        max_epochs = 10000
        batches_test_len = 0 
        test_batch_size = None
        if batches_test is not None:
            for batch in batches_test:
                batches_test_len += 1
                if test_batch_size is None: 
                    test_batch_size = len(batch)
        do_testing = batches_test_len > 0
        if verbose:
            if do_testing: 
                print(f'using a test set of {batches_test_len} batches (BS={test_batch_size})')
            else:
                print(f'no test set supplied')
                
        self.converged = False
        for epoch in range(max_epochs if until_convergence else epochs): 
            self.epoch = epoch
            for batch in batches_train:
                time_start = time.time()
                loss_last = self.train_batch(batch, optim)
                if loss_last is None: 
                    if verbose: 
                        print('Skipping empty batch')
                    continue
                self.batch_i += 1
                self.loss_accum.append(loss_last)
                duration = time.time() - time_start
                self.time_accum.append(duration)

                if print_every is not None and self.batch_i % print_every == 0:
                    last_accum_loss = self._record_train_loss(verbose)
                    if not do_testing: 
                        self._try_record_best_loss(last_accum_loss, verbose, min_improvement)

                if do_testing and test_every is not None and self.batch_i % test_every == 0:
                    last_test_loss = self._record_test_loss(batches_test, verbose)
                    self._try_record_best_loss(last_test_loss, verbose, min_improvement)

                if until_convergence and self.epochs_since_new_best >= patience:
                    print(f"converged after {epoch} epochs (lowest achieved loss: {self.best_loss:.4f})")
                    self.converged = True; break
            if self.converged: break
                
            if print_every is None:
                last_accum_loss = self._record_train_loss(verbose)
                if not do_testing: 
                    self._try_record_best_loss(last_accum_loss, verbose, min_improvement)                

            if do_testing and test_every is None:
                last_test_loss = self._record_test_loss(batches_test, verbose)
                self._try_record_best_loss(last_test_loss, verbose, min_improvement)

            if until_convergence and self.epochs_since_new_best >= patience:
                print(f"converged after {epoch} epochs (lowest achieved loss: {self.best_loss:.4f})")
                self.converged = True; break
            
        fpath = tmp_fname if self.save_best is None else self.save_best
        if os.path.isfile(fpath):
            if verbose:
                print("loading best checkpoint")
            self.load_state_dict(torch.load(fpath))
            #if save_best is not None: 
            #    shutil.copyfile(tmp_fname, save_best)
        if os.path.isfile(tmp_fname):
            os.remove(tmp_fname)
            
        return self.losses_train, self.losses_test, optim

    def train_batches_from_dict(
        self, 
        batches_train:Iterator, 
        params: dict, 
        batches_test:Iterator=None, 
        optim:torch.optim.Optimizer=None,
        ) -> Union[Iterator, Iterator, torch.optim.Optimizer]:

        epochs = params.get('epochs', -1)
        lr = params.get('lr', 1e-3)
        test_every = params.get('test_every', None)
        print_every = params.get('print_every', None)
        verbose = params.get('verbose', True)
        patience = params.get('patience', 5)
        min_improvement = params.get('min_improvement', 0)
        save_best = params.get('save_best', None)
        return self.train_batches(
            batches_train=batches_train, 
            epochs=epochs, 
            lr=lr,
            batches_test=batches_test, 
            test_every=test_every,
            print_every=print_every, 
            verbose=verbose, 
            optim=optim, 
            patience=patience, 
            min_improvement=min_improvement, 
            save_best=save_best,)

class TokenProjectorLR(ReconstructionModel): 
    dim_token:int 

    def __init__(self, dim_token:int):
        super().__init__()
        self.dim_token=dim_token
        self.encoder = nn.Linear(dim_token, dim_token)
        self.decoder = nn.Linear(dim_token*2, 1)

    def forward(self, batch): 
        u, v, y = zip(*batch)
        u = T(np.array(u)).float()
        v = T(np.array(v)).float()
        y = T(np.array(y)).float()
        u = self.encoder(u)
        v = self.encoder(v)
        uv = torch.hstack([u, v])
        y_hat = self.decoder(uv).squeeze(1)
        return y_hat, y

    def loss_fn(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y.float())

class TokenProjectorDot(ReconstructionModel): 
    dim_token:int 

    def __init__(self, dim_token:int):
        super().__init__()
        self.dim_token=dim_token
        self.encoder = nn.Linear(dim_token, dim_token)

    def forward(self, batch): 
        u, v, y = zip(*batch)
        u = T(np.array(u)).float()
        v = T(np.array(v)).float()
        y = T(np.array(y)).float()
        u = self.encoder(u)
        v = self.encoder(v)
        dot = torch.sum(u*v, dim=1)
        return dot, y

    def loss_fn(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y.float())

class TestTrafo(ReconstructionModel):
    # this is only to test if positional encodings work 
    def __init__(self, max_len=8, dim_token=8, num_layers=4, **kwargs):
        super().__init__()
        self.max_len=max_len

        self.pos_enc = PositionalEncoding(dim_token, max_len, False)
        encoder_layer = nn.TransformerEncoderLayer(dim_token, **kwargs)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(dim_token, max_len)

    def loss_fn(self, x, p):
        p = p.unsqueeze(-1).expand(x.shape[:-1])
        return F.cross_entropy(x.view(-1, x.shape[-1]), p.flatten())

    def forward(self, x):
        p = T(np.random.permutation(self.max_len)).long()
        x = self.pos_enc(x)
        x = x[p]
        x = self.transformer(x)
        x = self.proj(x)
        return x, p

class UniversalVAE(ReconstructionModel):
    dim_latent: int
    beta: float
    mask_rate: float
    mask_with: Union[int, float]
    batch_first: bool
    padding_token: Union[int, float]
    categorical: bool

    def __init__(
        self, 
        encoder:nn.Module, 
        decoder:nn.Module, 
        dim_encoder_out:int, 
        dim_latent:int, 
        categorical:bool,
        dim_decoder_in:int=-1, 
        beta:float=1.0,
        masked:bool=False,
        skipgram:bool=False,
        mask_with:Union[int, float]=0,
        batch_first:bool=False,
        padding_token:Union[int, float]=-1,
        mask_agg:str='mean', # mean, first, max
        masked_batch_refeed_rate:int=None,
        ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dim_latent = dim_latent
        self.beta = beta
        self.masked=masked
        self.skipgram=skipgram
        self.mask_with=mask_with
        self.batch_first=batch_first
        self.padding_token=padding_token
        self.mask_agg=mask_agg
        self.masked_batch_refeed_rate=masked_batch_refeed_rate
        self.categorical=categorical

        self.fc1 = nn.Linear(dim_encoder_out, dim_encoder_out//2)
        self.fc21 = nn.Linear(dim_encoder_out//2, dim_latent)
        self.fc22 = nn.Linear(dim_encoder_out//2, dim_latent)

        if dim_decoder_in == -1: 
            dim_decoder_in = dim_encoder_out
        self.fc3 = nn.Linear(dim_latent, dim_decoder_in)
        #self.fc3 = nn.Linear(dim_latent, dim_decoder_in//2)
        #self.fc4 = nn.Linear(dim_decoder_in//2, dim_decoder_in)

    def train_batch(self, batch, optim) -> Union[T, None]:
        if self.masked:
            sublosses = []
            if self.masked_batch_refeed_rate is not None and self.masked_batch_refeed_rate > 0: 
                for i in range(self.masked_batch_refeed_rate):
                    # refeed with random masking
                    sublosses.append(super().train_batch(batch, optim))
            else: 
                # refeed position by position 
                positions = list(range(self.seq_dim_size(batch)))
                random.shuffle(positions)
                for position in positions: 
                    mask = self.generate_mask(batch, position, True)
                    subloss=super().train_batch(batch, optim, kwargs={'mask_ext':mask})
                    sublosses.append(subloss)
            loss = float(np.mean(sublosses))
        else: 
            loss = super().train_batch(batch, optim)
        return loss

    def encode(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.encoder(x)
        enc = self.fc1(enc)
        mean = self.fc21(enc)
        logvar = self.fc22(enc)
        return mean, logvar

    def decode(self, z):
        dec = self.fc3(z)
        #dec = F.relu(dec)
        #dec = self.fc4(dec)
        dec = self.decoder(dec)
        return dec

    def reconstruction_error(self, x_pred, x):
        if self.categorical: 
            return F.cross_entropy(x_pred.view(x.shape.numel(), -1), x.flatten(), reduction='mean')
        else:
            return F.mse_loss(x_pred, x, reduction='mean')

    def reparameterize(self, mean, logvar):
        std = torch.exp(.5*logvar)
        eps = torch.rand_like(std)*2-1
        return mean + eps*std

    def seq_dim(self) -> int: 
        return 1 if self.batch_first else 0

    def batch_dim(self) -> int: 
        return 0 if self.batch_first else 1

    def seq_dim_size(self, x:T) -> int:
        return x.shape[self.seq_dim()]

    def batch_dim_size(self, x:T) -> int:
        return x.shape[self.batch_dim()]

    def unpadded(self, x: T):
        if self.categorical: 
            is_not_padding = (x != self.padding_token)
        else:
            is_not_padding = torch.abs(x.mean(dim=-1) - self.padding_token) > 1e-8
        return is_not_padding

    def unpadded_range(self, x:T):
        is_not_padding = self.unpadded(x)
        res = is_not_padding.sum(dim=self.seq_dim())
        return res

    def generate_mask(self, x: T, pos=None, force_pos:bool=False) -> Tuple[T, T]: 
        if pos is None: 
            pos = torch.randint(0, self.seq_dim_size(x), (self.batch_dim_size(x),))
        elif type(pos) == int: 
            pos = T([pos for _ in range(self.batch_dim_size(x))]).long()
        else: 
            assert len(pos) == self.batch_dim_size(x)
        
        ranges = self.unpadded_range(x)
        last_unpadded_idx = ranges - 1
        if force_pos: 
            mask_idx_along_seq = pos
        else:    
            mask_idx_along_seq = torch.min(last_unpadded_idx, pos)
            mask_idx_along_seq = torch.max(mask_idx_along_seq, torch.zeros_like(mask_idx_along_seq).long())

        return mask_idx_along_seq
        
    def agg_over_time(self, mean:T, logvar:T)->T:
        if self.mask_agg=='max':
            mean, _ = mean.max(dim=self.seq_dim())
            logvar, _ = logvar.max(dim=self.seq_dim())
        elif self.mask_agg=='mean':
            mean = mean.mean(dim=self.seq_dim())
            logvar = logvar.mean(dim=self.seq_dim())
        else: 
            if self.batch_first: 
                mean = mean[:, 0]
                logvar = logvar[:, 0]
            else: 
                mean = mean[0]
                logvar = logvar[0]
        return mean, logvar

    def forward_normal(self, x:T):
        # encode 
        mean, logvar = self.encode(x)
        # reparameterize and decode
        if self.beta == 0: # if beta=0 then KLD term will not contribute to the loss anyways
            z = mean
        else:
            z = self.reparameterize(mean, logvar)
        x_pred = self.decode(z)
        return x_pred, x, mean, logvar, z

    def forward_mask(self, x:T, mask_ext:T=None):
        # "skipgram" version
        if mask_ext is None: 
            mask = self.generate_mask(x)
        else: 
            mask = mask_ext
        assert len(mask) == self.batch_dim_size(x)
    
        contexts = x.clone()
        if self.batch_first: 
            pivots = x[np.arange(self.batch_dim_size(x)), mask].clone() # index the masked tokens (later to be predicted and compared against)
            contexts[np.arange(self.batch_dim_size(x)), mask] = self.mask_with
        else: 
            pivots = x[mask, np.arange(self.batch_dim_size(x))].clone()
            contexts[mask, np.arange(self.batch_dim_size(x))] = self.mask_with
    
        # encode 
        if self.skipgram: 
            mean, logvar = self.encode(pivots.unsqueeze(self.seq_dim()))
        else: 
            mean, logvar = self.encode(contexts)
            # get rid of time dimension to produce representation for the masked token
            mean, logvar = self.agg_over_time(mean, logvar)
            mean = mean.unsqueeze(self.seq_dim())
            logvar = logvar.unsqueeze(self.seq_dim())

        # reparameterize and decode
        if self.beta == 0: # if beta=0 then KLD term will not contribute to the loss anyways
            z = mean
        else:
            z = self.reparameterize(mean, logvar)
        x_pred = self.decode(z)

        # squeeze again so that the output can be readily used as a representation for the masked token 
        x_pred = x_pred.squeeze(dim=self.seq_dim())
        z = z.squeeze(dim=self.seq_dim())
        mean = mean.squeeze(dim=self.seq_dim())
        logvar = logvar.squeeze(dim=self.seq_dim())

        if self.skipgram: 
            # expand to compare against multiple words in context
            exp_shape = (self.seq_dim_size(x), *x_pred.shape)
            x_pred = x_pred.unsqueeze(0).expand(exp_shape).contiguous()
            if self.batch_first: 
                x_pred = x_pred.transpose(0,1)
            return x_pred, contexts, mean, logvar, z
        else: 
            return x_pred, pivots, mean, logvar, z
            
    def forward(self, x:T, mask_ext:T=None) -> Tuple[T, T, T, T, T, T]:
        if self.masked: 
            return self.forward_mask(x, mask_ext)
        else: 
            return self.forward_normal(x)

    def loss_fn(self, x_pred:T, x:T, mean:T, logvar:T, z:T) -> Union[T, None]:
        if self.masked: 
            BCE = self.reconstruction_error(x_pred, x)
        else: 
            # compute loss between all non-padding elements in x and x_pred
            # dont move this filtering procedure to the forward_normal method, so that all (inc. padding) embeddings are outputted
            is_not_padding = self.unpadded(x)
            logvar = logvar[is_not_padding]
            mean = mean[is_not_padding]
            BCE = self.reconstruction_error(x_pred[is_not_padding], x[is_not_padding])
        
        BCE = BCE.mean()

        # KL divergence from prior on z
        KLD = -.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        elbo = BCE + (KLD * self.beta)
        return elbo

def prior(K, alpha):
    """
    Prior for the model.
    :K: number of categories
    :alpha: Hyper param of Dir
    :return: mean and variance tensors
    """
    # ラプラス近似で正規分布に近似
    # Approximate to normal distribution using Laplace approximation
    a = torch.Tensor(1, K).float().fill_(alpha)
    mean = a.log().t() - a.log().mean(1)
    var = ((1 - 2.0 / K) * a.reciprocal()).t() + (1.0 / K ** 2) * a.reciprocal().sum(1)
    return mean.t(), var.t() # Parameters of prior distribution after approximation

class DirichletVAE(UniversalVAE):
    """ Adopted from https://github.com/is0383kk/Dirichlet-VAE """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Dir prior
        self.prior_mean, self.prior_var = map(nn.Parameter, prior(self.dim_latent, 0.3)) # 0.3 is a hyper param of Dirichlet distribution
        self.prior_logvar = nn.Parameter(self.prior_var.log())
        self.prior_mean.requires_grad = False
        self.prior_var.requires_grad = False
        self.prior_logvar.requires_grad = False

    def encode(self, x):
        conv = self.encoder(x)
        #h1 = self.fc1(conv.view(-1, 1024)) # not sure if this view is necessary here.
        h1 = self.fc1(conv)
        return self.fc21(h1), self.fc22(h1)

    def decode(self, gauss_z):
        dir_z = F.softmax(gauss_z,dim=1) 
        # This variable (z) can be treated as a variable that follows a Dirichlet distribution (a variable that can be interpreted as a probability that the sum is 1)
        # Use the Softmax function to satisfy the simplex constraint
        # シンプレックス制約を満たすようにソフトマックス関数を使用
        #z = dir_z.argmax(-1)
        #z_emb = self.decoder_emb(z)
        
        h3 = self.relu(self.fc3(dir_z))
        deconv_input = self.fc4(h3)
        #deconv_input = deconv_input.view(-1,1024,1,1)
        return self.decoder(deconv_input)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        gauss_z = self.reparameterize(mu, logvar) 
        # gause_z is a variable that follows a multivariate normal distribution
        # Inputting gause_z into softmax func yields a random variable that follows a Dirichlet distribution (Softmax func are used in decoder)
        dir_z = F.softmax(gauss_z,dim=1) # This variable follows a Dirichlet distribution
        x_pred = self.decode(gauss_z)
        return x_pred, x, mu, logvar

    def reconstruction_error(self, x_pred, x):
        raise Exception('implement me pls')

    # Reconstruction + KL divergence losses summed over all elements and batch    
    def loss_fn(self, x_pred, x, mu, logvar):
        K = 10
        beta = 1.0
        BCE = self.reconstruction_error(x_pred, x)
        #BCE = F.cross_entropy(x_pred.view(x.shape.numel(), -1), x.flatten(), reduction='sum')
        # ディリクレ事前分布と変分事後分布とのKLを計算
        # Calculating KL with Dirichlet prior and variational posterior distributions
        # Original paper:"Autoencodeing variational inference for topic model"-https://arxiv.org/pdf/1703.01488
        prior_mean = self.prior_mean.expand_as(mu)
        prior_var = self.prior_var.expand_as(logvar)
        prior_logvar = self.prior_logvar.expand_as(logvar)
        var_division = logvar.exp() / prior_var # Σ_0 / Σ_1
        diff = mu - prior_mean # μ_１ - μ_0
        diff_term = diff *diff / prior_var # (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1
        logvar_division = prior_logvar - logvar # log|Σ_1| - log|Σ_0| = log(|Σ_1|/|Σ_2|)
        # KL
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(-1) - K)        
        loss = BCE + KLD
        loss = loss.mean()
        return loss

class CategoricalDirichletVAE(DirichletVAE):
    def reconstruction_error(self, x_pred, x):
        return F.cross_entropy(x_pred.view(x.shape.numel(), -1), x.flatten(), reduction='sum')

class ContinuousDirichletVAE(DirichletVAE):
    def reconstruction_error(self, x_pred, x):
        return F.mse_loss(x_pred, x, reduction='sum')
        
class LambdaModule(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x)

def init_embedding(nin:int, nout=2, rang=10, typ='equidistant', seed=None):
    res = nn.Embedding(nin, nout) # default initialization is gaussian
    if typ=='equidistant':
        steps=int(np.ceil(nin**(1/nout)))
        singles = [torch.linspace(-(rang/2), rang/2, steps=steps) for _ in range(nout)]
        axes = torch.meshgrid(*singles, indexing='ij')
        stacked = torch.stack(axes)
        weights = stacked.transpose(0,2).reshape(-1, nout).contiguous()
        np.random.seed(seed)
        weights = weights[np.random.choice(np.arange(weights.shape[0]), replace=False, size=nin)]
        res.weight = nn.Parameter(weights)
        res.weight.requires_grad=False
    elif typ=='uniform':
        torch.manual_seed(seed)
        res.weight.data.uniform_(-(rang/2), rang/2)
        res.weight.requires_grad=False
    return res

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#emb = init_embedding(100, 3, typ='equidistant')
#ax.scatter(*emb.weight.T.detach())