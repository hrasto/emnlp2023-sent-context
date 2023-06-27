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
from re import sub
from collections import deque

tmp_fname = '.tmp.pt'

def camel_case(s):
  s = sub(r"(_|-)+", " ", s).title().replace(" ", "")
  return ''.join([s[0].lower(), s[1:]])

def format_as_filename(params): 
    def val2str(val):
        if type(val) == bool: 
            return 'Y' if val else 'N'
        if type(val) == float: 
            return f'{val:.2E}'
        if type(val) == int: 
            return str(val)
        return val
    parts = [f'{camel_case(key)}={val2str(val)}' for key, val in params.items()]
    return '_'.join(parts)

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
        self.hyperparams=kwargs
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

            if self.checkpoints is not None: 
                if len(self.checkpoints) == self.checkpoints.maxlen:
                    to_delete = self.checkpoints.popleft()
                    os.remove(to_delete)
                info = format_as_filename({'epoch':self.epoch, 'batch':self.batch_i, 'loss':self.best_loss})
                cpt = f'{self.save_best}.{info}.cpt'
                self.checkpoints.append(cpt)
                torch.save(self.state_dict(), cpt)
            
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
        save_best:str=None,
        save_last=0,
        rehearsal_run=False) -> Union[Iterator, Iterator, torch.optim.Optimizer]:
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

        if save_last > 0: 
            self.checkpoints = deque(maxlen=save_last)
        else: 
            self.checkpoints=None

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

                if (until_convergence and self.epochs_since_new_best >= patience) or (rehearsal_run and self.best_loss < np.inf):
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

            if (until_convergence and self.epochs_since_new_best >= patience) or (rehearsal_run and self.best_loss < np.inf):
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
        save_last = params.get('save_last', 0)
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
            save_best=save_best,
            save_last=save_last,)

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

class CNNEncoder(nn.Module):
    def __init__(self, dim_token:int, n_conv_filters:int, dropout:float, n_fc:int) -> None:
        super().__init__()
        self.dim_token=dim_token
        self.n_conv_filters=n_conv_filters
        self.dropout=dropout
        self.n_fc=n_fc
        self.cnn1 = nn.Sequential(nn.Conv1d(in_channels=dim_token, out_channels=n_conv_filters, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2))
        self.cnn2 = nn.Sequential(nn.Conv1d(in_channels=n_conv_filters, out_channels=n_conv_filters, kernel_size=3, padding=1), nn.ReLU())
        self.cnn3 = nn.Sequential(nn.Conv1d(in_channels=n_conv_filters, out_channels=n_conv_filters, kernel_size=3, padding=1), nn.ReLU())
        self.cnn4 = nn.Sequential(nn.Conv1d(in_channels=n_conv_filters, out_channels=n_conv_filters, kernel_size=3, padding=0), nn.ReLU(), nn.MaxPool1d(2))
        self.fc1 = nn.Sequential(nn.Linear(n_conv_filters*4, n_fc), nn.Dropout(dropout))
        self.fc2 = nn.Sequential(nn.Linear(n_fc, n_fc), nn.Dropout(dropout)),
        self.fc3 = nn.Linear(n_fc, dim_token), # (batch, dim_out)

    def forward(self, x:T):
        # input should be (batch_size, dim_token, sentence_len)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = x.view((-1, self.n_conv_filters*4))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x # should be (batch_size, dim_token)

class CNNDecoder(nn.Module):
    def __init__(self, dim_token:int, n_conv_filters:int, dropout:float, n_fc:int) -> None:
        super().__init__()
        self.dim_token=dim_token
        self.n_conv_filters=n_conv_filters
        self.dropout=dropout
        self.n_fc=n_fc

        self.fc1 = nn.Sequential(nn.Linear(dim_token, n_fc), nn.Dropout(dropout))
        self.fc2 = nn.Sequential(nn.Linear(n_fc, n_fc), nn.Dropout(dropout))
        self.fc3 = nn.Linear(n_fc, n_conv_filters*4)
        #LambdaModule(lambda x: x.view((-1, n_conv_filters, 4))),
        self.cnn1 = nn.Sequential(nn.ConvTranspose1d(in_channels=n_conv_filters, out_channels=n_conv_filters, kernel_size=5, padding=0), nn.ReLU()),
        self.cnn2 = nn.Sequential(nn.ConvTranspose1d(in_channels=n_conv_filters, out_channels=n_conv_filters, kernel_size=5, padding=0), nn.ReLU()),
        self.cnn3 = nn.Sequential(nn.ConvTranspose1d(in_channels=n_conv_filters, out_channels=n_conv_filters, kernel_size=5, padding=0), nn.ReLU()),
        self.cnn4 = nn.Sequential(nn.ConvTranspose1d(in_channels=n_conv_filters, out_channels=dim_token, kernel_size=5, padding=0), nn.ReLU()),
        
    def forward(self, x:T):
        # input should be (batch_size, dim_token)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.view((-1, self.n_conv_filters, 4))
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        return x # should be (batch_size, dim_token, sentence_len)

class SentenceProjector(ReconstructionModel):
    dim_token:int 

    def __init__(self, dim_token:int, n_conv_filters=128, dropout=.5, n_fc=256):
        super().__init__()
        self.dim_token=dim_token
        self.encoder = CNNEncoder(dim_token, n_conv_filters, dropout, n_fc)

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

class SentenceVAE(ReconstructionModel):
    dim_latent: int
    beta: float
    categorical: bool

    def __init__(
        self, 
        encoder:nn.Module, 
        decoder:nn.Module, 
        dim_token:int, 
        dim_latent:int, 
        cnn_params:dict,
        beta:float=1.0,
        ):
        super().__init__()
        self.encoder = CNNEncoder(**cnn_params)
        self.decoder = CNNDecoder(**cnn_params)
        self.dim_latent = dim_latent
        self.beta = beta

        self.fc1 = nn.Linear(dim_token, dim_latent)
        self.fc2 = nn.Linear(dim_token, dim_latent)
        self.fc3 = nn.Linear(dim_latent, dim_token)

    def encode(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.encoder(x)
        mean = self.fc1(enc)
        logvar = self.fc2(enc)
        return mean, logvar

    def decode(self, z):
        dec = self.fc3(z)
        dec = self.decoder(dec)
        return dec

    def reconstruction_error(self, x_pred, x):
        return F.mse_loss(x_pred, x, reduction='mean')

    def reparameterize(self, mean, logvar):
        std = torch.exp(.5*logvar)
        eps = torch.rand_like(std)*2-1
        return mean + eps*std

    def forward(self, x:T):
        # encode 
        mean, logvar = self.encode(x)
        # reparameterize and decode
        if self.beta == 0: # if beta=0 then this is a regular autoencoder
            z = mean
        else:
            z = self.reparameterize(mean, logvar)
        x_pred = self.decode(z)
        return x_pred, x, mean, logvar, z

    def loss_fn(self, x_pred:T, x:T, mean:T, logvar:T, z:T) -> Union[T, None]:
        BCE = self.reconstruction_error(x_pred, x)        
        BCE = BCE.mean()
        # KL divergence from prior on z
        KLD = -.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        elbo = BCE + (KLD * self.beta)
        return elbo

class LambdaModule(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x)

class SequenceSG(ReconstructionModel):
    def __init__(self, dim_token, dim_latent, hyperparams={}, emb_layer=None):
        super().__init__()
        self.emb_layer = emb_layer
        self.encoder = LSTMEncoder(dim_token, dim_latent, **hyperparams)
        decoder_lstm = LSTMEncoder(dim_token+dim_latent, dim_token, **hyperparams)
        self.decoder = TurboSequencer(decoder_lstm, dim_token, dim_latent)

    def forward(self, batch):
        pivot, neighbor = zip(*batch)
        if self.emb_layer is None: 
            pivot = T(np.array(pivot)).float()
            neighbor = T(np.array(neighbor)).float()
        else: 
            pivot = T(np.array(pivot)).long()
            neighbor = T(np.array(neighbor)).long()
            pivot = self.emb_layer(pivot)
            neighbor = self.emb_layer(neighbor)
        # transpose to sequence-first shape
        pivot = pivot.transpose(0, 1).contiguous() 
        neighbor = neighbor.transpose(0, 1).contiguous()
        lat = self.encoder(pivot)
        neighbor_hat = self.decoder.decode(
            x_static=lat, 
            teacher_force_y=neighbor,
            #decoding_steps=
            )
        return neighbor_hat, neighbor
    
    def loss_fn(self, neighbor_hat, neighbor):
        return F.mse_loss(neighbor_hat, neighbor, reduction='mean')

class TokenSG(ReconstructionModel):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Linear(dim, dim)
        self.decoder = nn.Linear(dim, dim)

    def forward(self, batch):
        pivot, neighbor = zip(*batch)
        pivot = T(np.array(pivot)).float()
        neighbor = T(np.array(neighbor)).float()

        lat = self.encoder(pivot)
        neighbor_hat = self.decoder(lat)
        return neighbor_hat, neighbor
    
    def loss_fn(self, neighbor_hat, neighbor):
        return F.mse_loss(neighbor_hat, neighbor, reduction='mean')


def init_embedding(nin:int, nout=2, rang=10, typ='equidistant', seed=None):
    res = nn.Embedding(nin, nout) # default initialization is gaussian
    if typ=='equidistant':
        steps=int(np.ceil(nin**(1/nout)))
        singles = [torch.linspace(-(rang/2), rang/2, steps=steps) for _ in range(nout)]
        axes = torch.meshgrid(*singles, indexing='ij')
        stacked = torch.stack(axes)
        weights = stacked.transpose(0,-1).reshape(-1, nout).contiguous()
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