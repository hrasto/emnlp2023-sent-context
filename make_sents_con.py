from cgi import test
from curses.ascii import ctrl
import os
import random
from segmenters.structure import StructuredCorpus
import segmenters.iterator as it
import torch
import torch.nn as nn
from torch import Tensor as T
import torch.nn.functional as F
from module import ReconstructionModel, TokenProjectorDot, TokenProjectorLR
import numpy as np

seed=1

random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

dirs = [
    'swda_mfs_20w',
    'swda_mfs_100w',
]

context_size=2
neg_samples=5
bs=256

"""
tp = TokenProjectorDot(8)
u = torch.rand((16, 8))
v = torch.rand((16, 8))
y = torch.randint(0, 2, (16,))
yh, y = tp(u, v, y)
tp.loss_fn(yh, y)
"""

def make_skipgrams(x: np.array, segmentation:list, context_size:int, neg_samples):
    seg_simple = np.array([-1] + segmentation + [x.shape[0]-1]) + 1
    for left, right in zip(seg_simple[:-1], seg_simple[1:]): 
        for pivot in range(left+context_size, right-context_size):
            for i in range(1, context_size+1):
                yield (x[pivot], x[pivot+i], 1)
                yield (x[pivot], x[pivot-i], 1)
            negs = set(np.random.choice(x.shape[0], neg_samples, replace=False))
            negs = negs.difference(set(range(left, right)))
            for neg in negs: 
                yield(x[pivot], x[neg], 0)

for dirname in dirs: 
    print(dirname)
    ctrain = StructuredCorpus.load(f'{dirname}/corpus/train')
    ctest = StructuredCorpus.load(f'{dirname}/corpus/test')
    cdev = StructuredCorpus.load(f'{dirname}/corpus/dev')

    if not os.path.isdir(f'{dirname}/sents_con'):
        os.mkdir(f'{dirname}/sents_con')
    if not os.path.isdir(f'{dirname}/models'):
        os.mkdir(f'{dirname}/models')
    
    for name in os.listdir(f'{dirname}/sents_iso'):
        """
        """
        x_train = np.load(f'{dirname}/sents_iso/{name}/train.npy')
        segmentation_train = list(ctrain.derive_segment_boundaries('conversation_no', 'default'))
        skipgrams_train = it.RestartableCallableIterator(make_skipgrams, fn_args=[x_train, segmentation_train, context_size, neg_samples])
        batches_train = it.RestartableBatchIterator(skipgrams_train, bs)

        x_dev = np.load(f'{dirname}/sents_iso/{name}/dev.npy')
        segmentation_dev = list(cdev.derive_segment_boundaries('conversation_no', 'default'))
        skipgrams_dev = it.RestartableCallableIterator(make_skipgrams, fn_args=[x_dev, segmentation_dev, context_size, neg_samples])
        batches_dev = it.RestartableBatchIterator(skipgrams_dev, bs)

        tp = TokenProjectorDot(x_dev.shape[1])
        """
        """
        tp.train_batches(
            batches_train=batches_train, 
            batches_test=batches_dev, 
            print_every=200,
            test_every=1000,
            epochs=-1, 
            lr=1e-3,
            patience=5,
            min_improvement=.0, 
            save_best=f'{dirname}/models/{name}_dot.pt'
        )

        if not os.path.isdir(f'{dirname}/sents_con/{name}_dot'): 
            os.mkdir(f'{dirname}/sents_con/{name}_dot')

        x_test = np.load(f'{dirname}/sents_iso/{name}/train.npy')
        test_encoded = tp.encoder(T(x_test).float()).detach().numpy()
        dev_encoded = tp.encoder(T(x_dev).float()).detach().numpy()
        np.save(f'{dirname}/sents_con/{name}_dot/test.npy', test_encoded)
        np.save(f'{dirname}/sents_con/{name}_dot/dev.npy', dev_encoded)
        np.save(f'{dirname}/sents_con/{name}_dot/train.npy', dev_encoded)

        tp = TokenProjectorLR(x_dev.shape[1])
        """
        """
        tp.train_batches(
            batches_train=batches_train, 
            batches_test=batches_dev, 
            print_every=200,
            test_every=1000,
            epochs=-1, 
            lr=1e-3,
            patience=5,
            min_improvement=.0, 
            save_best=f'{dirname}/models/{name}_lr.pt'
        )

        if not os.path.isdir(f'{dirname}/sents_con/{name}_lr'): 
            os.mkdir(f'{dirname}/sents_con/{name}_lr')

        test_encoded = tp.encoder(T(x_test).float()).detach().numpy()
        dev_encoded = tp.encoder(T(x_dev).float()).detach().numpy()
        np.save(f'{dirname}/sents_con/{name}_lr/test.npy', test_encoded)
        np.save(f'{dirname}/sents_con/{name}_lr/dev.npy', dev_encoded)
        np.save(f'{dirname}/sents_con/{name}_lr/train.npy', dev_encoded)