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
    #'swda_mfs_20w',
    'swda_mfs_100w',
]

def cut_and_pad_sequence(seq, max_words=20):
    seq = seq[:max_words]
    res = np.repeat(ctest.word_to_idx['<PAD>'], max_words)
    res[:len(seq)] = seq
    return res

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

for dim_latent in [6, 12, 25]:
    for dirname in dirs: 
        model_name = f'AE_dim={dim_latent}'
        print(dirname, model_name)
        ctrain = StructuredCorpus.load(f'{dirname}/corpus/train')
        ctest = StructuredCorpus.load(f'{dirname}/corpus/test')
        cdev = StructuredCorpus.load(f'{dirname}/corpus/dev')

        segmentation_train = list(ctrain.derive_segment_boundaries('conversation_no', 'default'))
        segmentation_dev = list(cdev.derive_segment_boundaries('conversation_no', 'default'))    

        idx_train = np.array([cut_and_pad_sequence(seq) for seq in ctrain.sequences])
        idx_dev = np.array([cut_and_pad_sequence(seq) for seq in cdev.sequences])
        idx_test = np.array([cut_and_pad_sequence(seq) for seq in ctest.sequences])

        duplicate = lambda x: (x, x)
        batches_train = it.RestartableMapIterator(idx_train, duplicate)
        batches_train = it.RestartableBatchIterator(batches_train, batch_size=batch_size)

        batches_dev = it.RestartableMapIterator(idx_dev[:2000], duplicate)
        batches_dev = it.RestartableBatchIterator(batches_dev, batch_size=batch_size*4)
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

        if not os.path.isdir(f'{dirname}/sents_iso'):
            os.mkdir(f'{dirname}/sents_iso')

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

        if not os.path.isdir(f'{dirname}/sents_iso/{model_name}'):
            os.mkdir(f'{dirname}/sents_iso/{model_name}')

        #np.save(f'{dirname}/sents_iso/{model_name}/train.npy', lsi_train)
        np.save(f'{dirname}/sents_iso/{model_name}/dev.npy', embs_dev)
        np.save(f'{dirname}/sents_iso/{model_name}/test.npy', embs_test)