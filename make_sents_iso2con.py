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

batch_size=512
training_params={
    'epochs':-1, 
    'lr':1e-3, 
    'print_every':10,
    'test_every':50,
    'patience':5,
    'min_improvement':.0,
}

for dirname in dirs: 
    if not os.path.isdir(f'{dirname}/sents_iso'):
        print(f'no isolated sentence representations found in directory {dirname}')
        continue

    ctrain = StructuredCorpus.load(f'{dirname}/corpus/train')
    ctest = StructuredCorpus.load(f'{dirname}/corpus/test')
    cdev = StructuredCorpus.load(f'{dirname}/corpus/dev')

    segmentation_train = list(ctrain.derive_segment_boundaries('conversation_no', 'default'))
    segmentation_dev = list(cdev.derive_segment_boundaries('conversation_no', 'default'))    

    for name in os.listdir(f'{dirname}/sents_iso'):
        if name[0] == '.' or name[:2] == 'AE': continue

        # dim_latent inferred from isolated sentence representation shape
        embs_train = np.load(f'{dirname}/sents_iso/{name}/train.npy')
        embs_dev = np.load(f'{dirname}/sents_iso/{name}/dev.npy')
        embs_test = np.load(f'{dirname}/sents_iso/{name}/test.npy')
        dim_latent = embs_train.shape[1]

        for context_size in [1, 2]:
            model_name = f'{name}_'+module.format_as_filename({'context':context_size,})
            print(dirname, model_name)

            batches_train = it.RestartableCallableIterator(make_skipgrams, fn_args=[embs_train, segmentation_train, context_size])
            batches_train = it.RestartableBatchIterator(batches_train, batch_size=batch_size)

            batches_dev = it.RestartableCallableIterator(make_skipgrams, fn_args=[embs_dev, segmentation_dev, context_size, 32*100])
            batches_dev = it.RestartableBatchIterator(batches_dev, batch_size=batch_size)
            #print(*zip(*next(iter(batches_dev))))

            # no embedding layer this time, as we already input sentence embeddings and only project them
            model = module.TokenSG(dim=dim_latent)
            
            if not os.path.isdir(f'{dirname}/models'):
                os.mkdir(f'{dirname}/models')

            if not os.path.isdir(f'{dirname}/models/{model_name}'):
                os.mkdir(f'{dirname}/models/{model_name}')
                
            losses_train, losses_test, _ = model.train_batches(
                batches_train=batches_train,
                batches_test=batches_dev,
                save_best=f'{dirname}/models/{model_name}/model.pt',
                save_last=30,
                #rehearsal_run=True,
                **training_params
            )
            meta = {
                'training_params': training_params, 
                'dim_latent': dim_latent,
                'dim_token': dim_latent,
                'batch_size': batch_size,
                'dirname': dirname,
                'model_name': model_name,
                'losses_train': losses_train,
                'losses_test': losses_test,
            }
            yaml.dump(meta, open(f'{dirname}/models/{model_name}/meta.yml', 'w'))

            if not os.path.isdir(f'{dirname}/sents_con'):
                os.mkdir(f'{dirname}/sents_con')

            batches_test = it.RestartableBatchIterator(list(embs_test), batch_size*4)
            batches_test = it.RestartableMapIterator(batches_test, lambda batch: T(batch).float())

            embs_test=[]
            for batch in batches_test: 
                #batch = emb_layer(batch)
                with torch.no_grad(): 
                    model.eval()
                    batch = model.encoder(batch)
                embs_test.append(batch)
            embs_test = torch.vstack(embs_test).detach().numpy()

            batches_dev = it.RestartableBatchIterator(list(embs_dev), batch_size*4)
            batches_dev = it.RestartableMapIterator(batches_dev, lambda batch: T(batch).float())

            embs_dev=[]
            for batch in batches_dev: 
                #batch = emb_layer(batch)
                with torch.no_grad(): 
                    model.eval()
                    batch = model.encoder(batch)
                embs_dev.append(batch)
            embs_dev = torch.vstack(embs_dev).detach().numpy()

            if not os.path.isdir(f'{dirname}/sents_con/{model_name}'):
                os.mkdir(f'{dirname}/sents_con/{model_name}')

            #np.save(f'{dirname}/sents_con/{model_name}/train.npy', lsi_train)
            np.save(f'{dirname}/sents_con/{model_name}/dev.npy', embs_dev)
            np.save(f'{dirname}/sents_con/{model_name}/test.npy', embs_test)