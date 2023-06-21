from segmenters.structure import StructuredCorpus
import segmenters.iterator as it
import module 
import yaml
from yaml.loader import SafeLoader

from gensim.models import TfidfModel, LsiModel, LdaMulticore, LdaModel
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors

import torch
import torch.nn as nn
from torch import Tensor as T
import torch.nn.functional as F

import numpy as np
import os

dirs = [
    'swda_mfs_20w',
]

def cut_and_pad_sequence(seq, max_words=20):
    seq = seq[:max_words]
    res = np.repeat(ctest.word_to_idx['<PAD>'], max_words)
    res[:len(seq)] = seq
    return res

def make_skipgrams(x: np.array, segmentation:list, context_size:int, limit=10000000):
    seg_simple = np.array([-1] + segmentation + [x.shape[0]-1]) + 1
    ct = 0
    for left, right in zip(seg_simple[:-1], seg_simple[1:]): 
        for pivot in range(left+context_size, right-context_size):
            for i in range(1, context_size+1):
                if ct >= limit: 
                    return
                yield (x[pivot], x[pivot+i])
                yield (x[pivot], x[pivot-i])
                ct+=2

dim_token = 8
batch_size=32
hyperparams = {
    'bidirectional': True, 
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': .1,
}
training_params={
    'epochs':-1, 
    'lr':3e-3, 
    'print_every':10,
    'test_every':100,
    'patience':5,
    'min_improvement':.0,
}

for context_size in [1, 2]:
    for dim_latent in [6, 12, 25]:
        for dirname in dirs: 
            model_name = 'SG_'+module.format_as_filename({'dim': dim_latent, 'context':context_size,})
            print(dirname, model_name)
            ctrain = StructuredCorpus.load(f'{dirname}/corpus/train')
            ctest = StructuredCorpus.load(f'{dirname}/corpus/test')
            cdev = StructuredCorpus.load(f'{dirname}/corpus/dev')

            segmentation_train = list(ctrain.derive_segment_boundaries('conversation_no', 'default'))
            segmentation_dev = list(cdev.derive_segment_boundaries('conversation_no', 'default'))    

            idx_train = np.array([cut_and_pad_sequence(seq) for seq in ctrain.sequences])
            idx_dev = np.array([cut_and_pad_sequence(seq) for seq in cdev.sequences])
            idx_test = np.array([cut_and_pad_sequence(seq) for seq in ctest.sequences])

            batches_train = it.RestartableCallableIterator(make_skipgrams, fn_args=[idx_train, segmentation_train, context_size])
            batches_train = it.RestartableBatchIterator(batches_train, batch_size=batch_size)

            batches_dev = it.RestartableCallableIterator(make_skipgrams, fn_args=[idx_dev, segmentation_dev, context_size, 32*100])
            batches_dev = it.RestartableBatchIterator(batches_dev, batch_size=batch_size)
            #print(*zip(*next(iter(batches_dev))))

            emb_layer = module.init_embedding(len(ctest), dim_token, seed=1)
            model_ae = module.SequenceSG(dim_token, dim_latent, hyperparams, emb_layer)
            
            if not os.path.isdir(f'{dirname}/models'):
                os.mkdir(f'{dirname}/models')

            if not os.path.isdir(f'{dirname}/models/{model_name}'):
                os.mkdir(f'{dirname}/models/{model_name}')
                
            losses_train, losses_test, _ = model_ae.train_batches(
                batches_train=batches_train,
                batches_test=batches_dev,
                save_best=f'{dirname}/models/{model_name}/model.pt',
                save_last=50,
                #rehearsal_run=True,
                **training_params
            )
            meta = {
                'training_params': training_params, 
                'hyperparams': hyperparams,
                'dim_latent': dim_latent,
                'dim_token': dim_token,
                'batch_size': batch_size,
                'dirname': dirname,
                'model_name': model_name,
                'losses_train': losses_train,
                'losses_test': losses_test,
            }
            yaml.dump(meta, open(f'{dirname}/models/{model_name}/meta.yml', 'w'))

            """
            if not os.path.isdir(f'{dirname}/sents_con'):
                os.mkdir(f'{dirname}/sents_con')

            batches_test = it.RestartableBatchIterator(list(idx_test), batch_size*4)
            batches_test = it.RestartableMapIterator(batches_test, lambda batch: T(batch).long().transpose(0, 1))

            embs_test=[]
            for batch in batches_test: 
                batch = emb_layer(batch)
                with torch.no_grad(): 
                    model_ae.eval()
                    batch = model_ae.encoder(batch)
                embs_test.append(batch)
            embs_test = torch.vstack(embs_test).detach().numpy()

            batches_dev = it.RestartableBatchIterator(list(idx_dev), batch_size*4)
            batches_dev = it.RestartableMapIterator(batches_dev, lambda batch: T(batch).long().transpose(0, 1))

            embs_dev=[]
            for batch in batches_dev: 
                batch = emb_layer(batch)
                with torch.no_grad(): 
                    model_ae.eval()
                    batch = model_ae.encoder(batch)
                embs_dev.append(batch)
            embs_dev = torch.vstack(embs_dev).detach().numpy()

            if not os.path.isdir(f'{dirname}/sents_con/{model_name}'):
                os.mkdir(f'{dirname}/sents_con/{model_name}')

            #np.save(f'{dirname}/sents_con/{model_name}/train.npy', lsi_train)
            np.save(f'{dirname}/sents_con/{model_name}/dev.npy', embs_dev)
            np.save(f'{dirname}/sents_con/{model_name}/test.npy', embs_test)
            """