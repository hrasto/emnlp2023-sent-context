from __future__ import annotations
from ast import Lambda
from multiprocessing.sharedctypes import Value
from typing import Tuple, Union
from unicodedata import category
import yaml
from yaml.loader import SafeLoader
import importlib
from typing import Iterator, List
from iterator import RestartableMapIterator
from module import *
import torch.nn as nn
from segmenters.structure import StructuredCorpus, Key
import segmenters.iterator as it
import torch
import torch.optim
import numpy as np
from torch import Tensor as T
from datetime import datetime as dt
from collections import deque
from sklearn.manifold import TSNE
import plotly.express as ply
import gensim.downloader
from gensim.models import KeyedVectors
from trial import trial

inference_batch_size=32
auto_context_size=-1

def set_random_state(seed):
    random.seed(seed)
    np.random.set_state(seed)
    torch.random.manual_seed(seed)

class LevelCompatibilityError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

default_params = {
    'dim_out': 2, 
    'dim_token': 8, 
    'max_context_size': 8, 
    'masking': None, 
    'encoder': 'trafo',
    'decoder': 'linear',
    'beta': .0,
}

def override_params(params:dict):
    return {**default_params, **params}

class AbstractLevel:
    dim_in: int
    dim_out: int
    max_context_size: int
    dim_out_concat: int
    segmentation_name: Key
    corpus: Union[StructuredCorpus, None]
    prev_lvl: AbstractLevel
    padding_token: Union[int, float]
    batch_first: bool
    categorical: bool

    def __init__(self, dim_in: int, dim_out: int, max_context_size: int, segmentation_name: str, corpus: StructuredCorpus, prev_lvl: AbstractLevel, padding_token: Union[int, float], batch_first:bool, categorical:bool) -> None:
        # combine keys so that segment boundaries match hierarchically
        if prev_lvl is not None: 
            if not StructuredCorpus.keys_overlap(prev_lvl.segmentation_name, segmentation_name):
                print(f'Note: keys from current lvl ({str(segmentation_name)}) dont overlap with keys from prev_lvl ({str(prev_lvl)})')

        if corpus is None: 
            print('Note: initialising level without a reference to a corpus')
        else: 
            try: 
                corpus[segmentation_name]
            except KeyError: 
                raise ValueError(f"segmentation name '{segmentation_name}' not found in supplied corpus")

        if max_context_size == auto_context_size: 
            if corpus is None: 
                raise ValueError('Must provide a max_context_size if corpus reference not provided')
            else: 
                if prev_lvl is None: 
                    boundaries = corpus.derive_segment_boundaries(segmentation_name)
                else: 
                    boundaries = corpus.derive_segment_boundaries(segmentation_name, prev_lvl.segmentation_name)
                boundaries = list(boundaries)
                max_context_size = get_max_segment_size(boundaries)
        assert max_context_size != auto_context_size

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.max_context_size = max_context_size 
        self.dim_out_concat = dim_out*max_context_size        
        self.padding_token=padding_token
        self.segmentation_name = segmentation_name
        self.corpus = corpus
        self.prev_lvl = prev_lvl
        self.batch_first=batch_first
        self.categorical=categorical
        self.corrupt_fn = lambda x: x

    def __str__(self): 
        if type(self.segmentation_name) == str: 
            name = self.segmentation_name
        else: 
            name = '&'.join(self.segmentation_name)
        if self.is_root(): 
            name = '[root] ' + name
        return name

    def is_root(self) -> bool:
        return self.prev_lvl == None

    def concat_batch(self, batch:T)->T: 
        if self.batch_first: 
            res = batch.reshape((batch.shape[0], -1))
        else: 
            res = batch.transpose(0,1).reshape((batch.shape[1], -1))
        if not self.is_root():
            assert res.shape[1] == self.prev_lvl.dim_out_concat
        return res
    
    """
    def get_input_from_level_below(self, corpus:StructuredCorpus=None, batch_size:int=inference_batch_size, sample:bool=False) -> Union[torch.Tensor, Iterator]:
        if self.is_root():
            if corpus is None: 
                if self.corpus is None: 
                    raise Exception("dont have any corpus to work with")
                else:
                    data = self.corpus.iter_flat() 
            else:
                data = corpus.iter_flat()
        else: 
            batches = self.prev_lvl.arrange_batches(batch_size, sample, corpus)
            batches_enc = it.RestartableMapIterator(batches, lambda batch: self.encode_batch(batch, sample))
            data = it.RestartableMapIterator(batches_enc, self.concat_batch)
        return data
    """

    def get_current_lvl_segmentation(self, corpus: StructuredCorpus = None) -> Iterator[int]:
        # derive segmentation boundaries with respect to the level below
        if corpus is None: 
            if self.corpus is None: 
                raise Exception("dont have any corpus to work with")
            else: 
                corpus = self.corpus
            
        if self.is_root():
            segmentation = corpus.derive_segment_boundaries(self.segmentation_name)
        else: 
            segmentation = corpus.derive_segment_boundaries(self.segmentation_name, self.prev_lvl.segmentation_name)
        return segmentation

    def _pad_segment(self, segment, is_tensor, pad_token): 
        padding_len = self.max_context_size - len(segment)
        if is_tensor: 
            segment = torch.vstack(segment)
            segment_padded = torch.zeros(self.max_context_size, self.dim_in)
            segment = segment[:min(segment.shape[0], self.max_context_size), :]
            segment_padded[:segment.shape[0], :] = segment
            segment = segment_padded
        else: 
            segment = segment[:min(len(segment), self.max_context_size)]
            segment += [[pad_token] for _ in range(padding_len)]
            segment = np.array(segment)
            segment = T(segment).long()
        return segment, padding_len

    def segment_and_pad(self, in_data: Iterator, segmentation: Iterator[int], return_padding_lens:bool=False) -> Iterator[torch.Tensor]:
        """Arranges the output of concatenate (of previous lvl) or flat tokens from corpus into batches and sequences."""    
        if self.padding_token is not None: 
            pad_token = self.padding_token
        else:
            pad_token = self.corpus.word_to_idx[self.corpus.unk_token]
        
        # do some things to find out things about these fucking things
        first_batch = next(iter(in_data))
        if type(first_batch) == T: 
            is_tensor = True
        else: 
            is_tensor = False

        if is_tensor and self.categorical: 
            print("Warning: processing real valued data in a categorical level")

        if is_tensor: 
            in_data = it.RestartableFlattenIterator(in_data) # flatten incoming batches
        # in_data is iterator of integers or tensors
        data_iter = iter(in_data)
    
        # iterate over segment boundary indices
        prev_boundary = None
        for boundary in segmentation: 
            # consume until boundary + 1
            if prev_boundary is None: 
                to_consume = boundary+1 
            else:
                to_consume = boundary-prev_boundary

            prev_boundary = boundary
            segment = []
            for _ in range(to_consume):
                try:
                    segment.append(next(data_iter))
                except StopIteration: 
                    print(f"Warning: segmentation boundary {boundary} exceeds iterator range; breaking from the loop. This is probably because you specified a segmentation that does not overlap with the lower level segmentation!")
                    break
            segment = self.corrupt_fn(segment)

            # pad this fucking segment
            segment, padding_len = self._pad_segment(segment, is_tensor, pad_token)
            if return_padding_lens:
                yield segment, padding_len
            else:
                yield segment

        # consume until end
        last_segment = []
        try: 
            while True: last_segment.append(next(data_iter))
        except StopIteration: 
            pass

        if len(last_segment) > 0: 
            last_segment = self.corrupt_fn(last_segment)
            last_segment, padding_len = self._pad_segment(last_segment, is_tensor, pad_token)
            if return_padding_lens:
                yield last_segment, padding_len
            else:
                yield last_segment
        else: 
            print("Warning: this should not happen")      

    def arrange_batches(self, batch_size: int=inference_batch_size, sample: bool=False, corpus: StructuredCorpus=None, return_padding_lens:bool=False) -> Tuple[RestartableMapIterator, Iterator]:
        # prepares data by encoding by lower levels and batchifying, outputting iterable of ready batches for current lvl. model
        # previously "get_batches_from_below" method
        if corpus is None: 
            corpus = self.corpus
        if corpus is None: 
            raise Exception("dont have any corpus to work with")

        if self.is_root():
            in_data = corpus.iter_flat()
        else: 
            # get input data from level below recursively (previously "transform" method)
            batches, padding_lens = self.prev_lvl.arrange_batches(batch_size, sample, corpus, return_padding_lens)
            batches_enc = it.RestartableMapIterator(batches, lambda batch: self.prev_lvl.encode_batch(batch, sample))
            if StructuredCorpus.keys_identical(self.segmentation_name, self.prev_lvl.segmentation_name):
                return batches_enc, padding_lens
            else:
                in_data = it.RestartableMapIterator(batches_enc, self.concat_batch)

        segmentation = it.RestartableCallableIterator(self.get_current_lvl_segmentation, fn_args=[corpus])
        if return_padding_lens: 
            # this one only works in-memory, so shouldnt be used with large datasets
            segments_and_lens = list(self.segment_and_pad(in_data, segmentation, True))
            segments, padding_lens = zip(*segments_and_lens)
        else:    
            segments = it.RestartableCallableIterator(self.segment_and_pad, fn_args=[in_data, segmentation, False])
            padding_lens = None

        # create batches (iterables of whatever we have in segments)
        batches = it.RestartableBatchIterator(segments, batch_size)

        # map batches with functions making them tensors and aligning them according to self.batch_first
        if self.is_root():
            # batch is a list of numpy arrays -> can call torch.Tensor on it
            if self.batch_first: 
                batch_map_fn = lambda batch: torch.stack(batch).squeeze(-1).long()
            else:
                batch_map_fn = lambda batch: torch.stack(batch).squeeze(-1).long().transpose(0, 1).contiguous()
        else: 
            # batch is a list of tensors -> call torch.stack on it
            if self.batch_first: 
                batch_map_fn = lambda batch: torch.stack(batch).float()
            else:
                batch_map_fn = lambda batch: torch.stack(batch).float().transpose(0, 1).contiguous()
                
        batches = it.RestartableMapIterator(batches, batch_map_fn)

        return batches, padding_lens

    def set_context_size(self, max_context_size:int):
        self.max_context_size = max_context_size
        self.dim_out_concat = self.dim_out*max_context_size

    """
    def transform(self, batch_size:int=inference_batch_size, sample:bool=False, corpus: StructuredCorpus=None) -> torch.Tensor:
        batches = self.arrange_batches(batch_size, sample, corpus)
        batches_enc = it.RestartableMapIterator(batches, lambda batch: self.encode_batch(batch, sample))
        out_data = it.RestartableMapIterator(batches_enc, self.concat_batch)
        return out_data
    """

    def encode(self, batch_size:int=inference_batch_size, sample:bool=False, corpus: StructuredCorpus=None):
        batches, padding_lens = self.arrange_batches(batch_size, sample, corpus, True)
        batches_enc = it.RestartableMapIterator(batches, lambda batch: self.encode_batch(batch, sample))
        if self.batch_first:
            stacked = torch.vstack(batches_enc)
        else:
            transposed = [batch.transpose(0, 1) for batch in batches_enc]
            stacked = torch.vstack(transposed)

        desegmented = np.empty((
            sum([self.max_context_size - max(0, l) for l in padding_lens]), 
            self.dim_out))

        entry = 0
        for segment_i, padding_len in enumerate(padding_lens): 
            for token_i in range(min(self.max_context_size, self.max_context_size-padding_len)): 
                desegmented[entry] = stacked[segment_i, token_i].detach().numpy()
                entry += 1

        if corpus is None: 
            corpus = self.corpus
        
        if self.prev_lvl is None: 
            prev_sname = None
        else: 
            prev_lvl = self.prev_lvl
            while prev_lvl is not None and StructuredCorpus.keys_identical(prev_lvl.segmentation_name, self.segmentation_name):
                prev_lvl = prev_lvl.prev_lvl
            if prev_lvl is None: 
                prev_sname = None
            else: 
                prev_sname = prev_lvl.segmentation_name
        
        segs_nested, labs_coarse = zip(*corpus.segment_wrt(self.segmentation_name, prev_sname))
        try: 
            segs_fine_trunc = [tuple(zip(*segs_and_labs))[0][:self.max_context_size] for segs_and_labs in segs_nested]
            labs_fine_trunc = [tuple(zip(*segs_and_labs))[1][:self.max_context_size] for segs_and_labs in segs_nested]
        except IndexError:
            raise Exception("check for empty sequences in your corpus, I can't work with those")

        return desegmented, segs_fine_trunc, labs_fine_trunc, labs_coarse

    def encode_batch(self, *args, **kwargs) -> T: 
        raise Exception("must implement me")

    def vis(self, tsne_n=2000, color='coarse', color_lab_idx=0, opacity=.3, fn=lambda x: x, context_size=2, max_segment_len=30):
        print('encoding...')
        embs, segs_fine_trunc, labs_fine_trunc, labs_coarse = self.encode()
        seqs, labf, labc, context_left, context_right = [], [], [], [], []

        def format_segment(segment, max_segment_len):
            if len(segment) > max_segment_len: 
                segment = segment[:(max_segment_len//2)-1]+'..'+segment[-((max_segment_len//2)+1):]
            return segment

        for seq, labs_fine, lab_coarse in zip(segs_fine_trunc, labs_fine_trunc, labs_coarse):
            seq_context = [self.corpus.decode_sent(item, stringify=True).replace(' ', '') for item in seq]
            seq_context = [format_segment(segment) for segment in seq_context]
            #seq_context = '|'.join(seq_context)
            #if len(seq_context) > context_len: 
            #    seq_context = seq_context[:context_len-2]+'..'
            for item, lf in zip(seq, labs_fine): 
                subseq = self.corpus.decode_sent(item, stringify=True).replace(' ', '')
                subseq = format_segment(subseq, max_segment_len*2)
                seqs.append(subseq)
                subseq_pos = seq_context.index(subseq)
                if subseq_pos > 0: 
                    context_start_idx = max(0, subseq_pos-context_size)
                    context_left.append('|'.join(seq_context[context_start_idx:subseq_pos]))
                else: 
                    context_left.append('-')
                
                if subseq_pos < len(seq_context)-1: 
                    context_end_idx = min(len(seq_context), subseq_pos+context_size+1)
                    context_right.append('|'.join(seq_context[subseq_pos+1:context_end_idx]))
                else:
                    context_right.append('-')
                labf.append(lf)
                labc.append(lab_coarse)

        n = embs.shape[0]
        if embs.shape[1] != 2: 
            print('tsne...')
            n = tsne_n
            embs = TSNE(n_components=2).fit_transform(embs[:n])

        if color == 'coarse': 
            colors = [l[color_lab_idx] if type(l) == tuple else l for l in labc]
        elif color == 'fine': 
            colors = [l[color_lab_idx] if type(l) == tuple else l for l in labf]
        else: 
            colors = [l[color_lab_idx] if type(l) == tuple else l for l in seqs]
        colors = [fn(c) for c in colors]

        print('plotting...')
        print(self.dirname)
        #title = lvl.dirname.split('_')
        #title = '\n'.join(title)
        title=self.dirname
        fig = ply.scatter(
            x=embs[:, 0], y=embs[:, 1], 
            color=colors[:n], 
            hover_data={'seq': seqs[:n], 'context_left': context_left[:n],'context_right': context_right[:n], 'lab_fine': labf[:n], 'lab_coarse': labc[:n]}, 
            opacity=opacity, 
            title=title)
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )
        return fig

class FixedEmbeddingLevel(AbstractLevel):
    def __init__(self, dim_in: int, dim_out: int, max_context_size: int, segmentation_name: str, corpus: StructuredCorpus, padding_token: Union[int, float], typ='equidistant', rang=10, seed=None, batch_first=False) -> None:
        super().__init__(dim_in, dim_out, max_context_size, segmentation_name, corpus, None, padding_token, batch_first, True)
        self.typ=typ
        self.emb = init_embedding(dim_in, dim_out, rang, typ, seed)

    def encode_batch(self, x, sample=False) -> T:
        return self.emb(x)

    def vis(self):
        embs = self.emb.weight.detach()
        hover_data = {}
        if self.corpus is not None and len(self.corpus.idx_to_word) == embs.shape[0]: 
            hover_data = {'word': self.corpus.idx_to_word}
        fig = ply.scatter(
            x=embs[:, 0], y=embs[:, 1],
            hover_data=hover_data)
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )
        return fig

class W2VEmbeddingLevel(AbstractLevel): 
    def __init__(self, dim_in: int, dim_out: int, max_context_size: int, segmentation_name: str, corpus: StructuredCorpus, padding_token: Union[int, float], batch_first: bool, pretrained='glove-twitter-25') -> None:
        if corpus is None: 
            raise ValueError('corpus must be provided for this type of level')

        super().__init__(dim_in, dim_out, max_context_size, segmentation_name, corpus, None, padding_token, batch_first, True)
        try: 
            self.w2v = KeyedVectors.load(pretrained)
            return 
        except ValueError: 
            pass
        
        self.w2v = gensim.downloader.load(pretrained)
        assert dim_out == self.w2v.vectors.shape[1]

    def my2wv(self, indices):
        return [self.w2v.key_to_index.get(self.corpus.idx_to_word[i], 0) for i in indices]

    def encode_batch(self, x, sample=False) -> T:
        orig_shape=x.shape
        flat_mine = x.flatten()
        flat_w2v = self.my2wv(flat_mine)
        flat_encoded = self.w2v[flat_w2v]
        encoded = T(flat_encoded).reshape((*orig_shape, -1))
        return encoded
        
class TrainableLevel(AbstractLevel):    
    model: UniversalVAE
    masking: str = None,
    mask_with: int
    mask_agg: str
    hyperparams: dict
    dirname: str
    beta:float

    def __init__(
        self, 
        dim_in: int, 
        dim_out: int, 
        dim_token: int, 
        segmentation_name: Key,
        max_context_size: int, 
        #encoder: str,
        #decoder: str,
        model: UniversalVAE, 
        padding_token: Union[int, float]=44, 
        masking: str = None,
        mask_with: Union[int, float] = 0,
        mask_agg: str='max',
        beta:float = .0,
        categorical: bool=False,
        batch_first: bool = False,
        corpus: StructuredCorpus = None,
        prev_lvl: AbstractLevel = None,
        hyperparams: dict={},
        dirname: str=None,
        ):
        super().__init__(dim_in, dim_out, max_context_size, segmentation_name, corpus, prev_lvl, padding_token, batch_first, categorical)
        self.mask_with=mask_with
        if dirname is not None and not os.path.isdir(dirname): 
            os.mkdir(dirname)
        self.dirname = dirname
        self.dim_token = dim_token
        self.masking=masking
        self.mask_agg=mask_agg
        self.beta=beta
        self.hyperparams = hyperparams        
        self.model = model
        if prev_lvl is None and not self.categorical: 
            print("Warning: initializing a real valued level that is a root level")        
        if prev_lvl is not None and self.categorical: 
            print("Warning: initializing a categorical level that is not a root level")

    def summarize(self):
        return {
            'dim_in': self.dim_in,
            'dim_out': self.dim_out,
            'dim_token': self.dim_token,
            'categorical': self.categorical,
            'segmentation_name': self.segmentation_name,
            'max_context_size': self.max_context_size,
            #'encoder': self.encoder,
            #'decoder': self.decoder,
            'padding_token': self.padding_token,
            'mask_with': self.mask_with,
            'mask_agg': self.mask_agg,
            'masking': self.masking,
            'beta': self.beta,
            'batch_first': self.batch_first,
            'hyperparams': self.hyperparams,
        }

    def save(self) -> AbstractLevel:
        if self.dirname is None: 
            raise Exception("must set a dirname before saving")
        if not os.path.isdir(self.dirname):
            os.mkdir(self.dirname)
        params = {'level_params': self.summarize()}
        with open(os.path.join(self.dirname, 'meta.yml'), 'w') as f:
            yaml.dump(params, f)
        torch.save(self.model.state_dict(), os.path.join(self.dirname, 'model.pt'))
        return self

    def load(dirname:str, corpus:StructuredCorpus=None, prev_lvl:AbstractLevel=None, run_name:str=None) -> TrainableLevel:
        with open(os.path.join(dirname, 'meta.yml'), 'r') as f: 
            file_content = yaml.load(f, Loader=SafeLoader)
        params = {}
        if 'level_params' in file_content:
            params = file_content['level_params']
        params['corpus'] = corpus
        params['prev_lvl'] = prev_lvl
        lvl = TrainableLevel(**params)
        lvl.dirname = dirname
        if run_name is None:
            model_path = os.path.join(dirname, 'model.pt')
        else: 
            model_path = os.path.join(dirname, run_name, 'model.pt')
        if os.path.isfile(model_path):
            lvl.model.load_state_dict(torch.load(model_path))
        else: 
            print(f'Note: could not find model weights at {model_path}')
        return lvl

    def fit(self, training_params: dict, batches, optim:torch.optim.Optimizer=None, batches_test=None, batch_refeed_rate:int=6, run_name:str='run', shuffle_sequences=False):
        self.model.masked_batch_refeed_rate = batch_refeed_rate
        if self.dirname is not None and not os.path.isdir(self.dirname):
            os.mkdir(self.dirname)

        if shuffle_sequences:
            batches = it.RestartableMapIterator(batches, self.shuffle_sequence)

        training_params = training_params.copy() # make sure this dictionary isn't manipulated during training
        
        save_best, run_meta_path, run_dir = None, None, None
        if self.dirname is not None: 
            run_dir = os.path.join(self.dirname, run_name)
            i = 2
            while os.path.isdir(run_dir):
                run_dir = os.path.join(self.dirname, f'{run_name}_v{i}')
                i += 1
            os.mkdir(run_dir)
            run_meta_path = os.path.join(run_dir, 'meta.yml')

            if 'save_best' not in training_params:
                training_params['save_best'] = os.path.join(run_dir, 'model.pt')

        losses_train, losses_test, optim = self.model.train_batches_from_dict(
            batches_train=batches, 
            params=training_params, 
            optim=optim, 
            batches_test=batches_test)

        if run_meta_path is not None: 
            with open(run_meta_path, 'w') as f: 
                meta = {
                    'corpus': self.corpus.dirname,
                    'timestamp': str(dt.now()),
                    'training_params': training_params,
                    'losses_train': losses_train,
                    'losses_test': losses_test,
                    'vocab_stats': self.corpus.statistics(),
                    'model_size': self.model.count_parameters()
                }
                yaml.dump(meta, f)
        
        if 'save_best' in training_params:
            # copy the freshly trained model to the 'default' location
            shutil.copy(training_params['save_best'], os.path.join(self.dirname, 'model.pt'))

        return losses_train, losses_test, optim

    def shuffle_sequence(self, batch):
        perm = list(range(self.max_context_size))
        random.shuffle(perm)
        if self.batch_first: 
            return batch[:, perm]
        else: 
            return batch[perm, :]

    def fit_corpus(self, training_params: dict, sample:bool=False, corpus: StructuredCorpus=None, optim:torch.optim.Optimizer=None, corpus_test: StructuredCorpus=None, batch_refeed_rate:int=6, run_name:str='run', shuffle_sequences=False):
        batch_size = training_params['batch_size']

        batches, _ = self.arrange_batches(batch_size, sample, corpus)
        if shuffle_sequences:
            batches = it.RestartableMapIterator(batches, self.shuffle_sequence)

        if corpus_test is not None: 
            batches_test, _ = self.arrange_batches(inference_batch_size, sample, corpus_test)
        else: 
            batches_test = None

        return self.fit(
            training_params=training_params, 
            batches=batches, 
            optim=optim, 
            batches_test=batches_test, 
            batch_refeed_rate=batch_refeed_rate, 
            run_name=run_name,
            shuffle_sequences=False)

    def predict(self, latent: Iterator) -> torch.Tensor:
        # token probabilities (within segmentations)
        pass

    def inverse_transform(self, latent: Iterator) -> StructuredCorpus:
        # reconstruct corpus as actual decoded tokens given vocab
        pass

    def encode_batch(self, batch: torch.Tensor, sample=False) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            if self.masking is not None:
                # iterate over positions, mask each one of them, stack the results
                seq_dim = self.model.seq_dim()
                res_per_pos = []
                for pos in range(batch.shape[seq_dim]):
                    mask = self.model.generate_mask(batch, pos, force_pos=True)
                    x_pred, x, mean, logvar, z = self.model.forward(batch, mask) # padding tokens will be ingored anyways in the desegmenting procedure of encode
                    if sample: 
                        if self.batch_first: 
                            res_per_pos.append(z)
                        else: 
                            res_per_pos.append(z)
                    else: 
                        if self.batch_first: 
                            res_per_pos.append(mean)
                        else: 
                            res_per_pos.append(mean)
                res = torch.stack(res_per_pos, dim=seq_dim)
            else:
                # no masking - take the full output as result 
                x_pred, x, mean, logvar, z = self.model.forward(batch)
                if sample: 
                    res = z
                else: 
                    res = mean
            
        res = res.detach()
        return res           

class Level(TrainableLevel):
    encoder: str
    decoder: str

    def __init__(
        self, 
        dim_in: int, 
        dim_out: int, 
        dim_token: int, 
        segmentation_name: Key,
        max_context_size: int, 
        encoder: str,
        decoder: str,
        padding_token: Union[int, float]=44, 
        masking: str = None,
        mask_with: Union[int, float] = 0,
        mask_agg: str='max',
        beta:float = .0,
        categorical: bool=False,
        batch_first: bool = False,
        corpus: StructuredCorpus = None,
        prev_lvl: AbstractLevel = None,
        hyperparams: dict={},
        dirname: str=None,
        ):        
        self.encoder = encoder
        self.decoder = decoder

        pos_enc_max_len = max(20, max_context_size) # prevent small positional encodings for cases when we want different context sizes for training and predicting
        first_enc_layer_type = nn.Embedding if categorical else nn.Linear

        if encoder in ['t', 'trafo', 'transformer']: 
            if masking == 'skipgram':
                print(f'Note: skipgram setup combined with transformer encoder might not be what you need')
            enc = nn.Sequential(
                first_enc_layer_type(dim_in, dim_token),
                TransformerModel(dim_token, max_len=pos_enc_max_len, aggregate=False, hyperparams=hyperparams),
            )
        elif encoder in ['nl', 'nonlinear']:
            dim_inter = min(512, 5*dim_token)
            enc = nn.Sequential(
                first_enc_layer_type(dim_in, dim_inter),
                nn.ReLU(),
                nn.Linear(dim_inter, dim_token)
            )
        else:
            enc = first_enc_layer_type(dim_in, dim_token)

        def pos_enc():
            if masking is None: 
                return PositionalEncoding(dim_token, pos_enc_max_len, batch_first)
            else: 
                return LambdaModule(lambda x: x) # masking models handle position encoding themselves

        if decoder in ['t', 'trafo', 'transformer']: 
            if masking == 'cbow':
                print(f'Note: cbow setup combined with transformer decoder might not be what you need')
            dec = nn.Sequential(
                TransformerModel(dim_token, max_len=pos_enc_max_len, aggregate=False, hyperparams=hyperparams),
                nn.Linear(dim_token, dim_in)
            )
        elif decoder in ['nl', 'nonlinear']:
            dim_inter = min(512, 5*dim_token)
            dec = nn.Sequential(
                pos_enc(),
                nn.Linear(dim_token, dim_inter),
                nn.ReLU(),
                nn.Linear(dim_inter, dim_in)
            )
        else:
            dec = nn.Sequential(
                pos_enc(),
                nn.Linear(dim_token, dim_in),
            )
                
        model = UniversalVAE(
            encoder=enc, decoder=dec,
            categorical=categorical, 
            dim_encoder_out=dim_token, dim_decoder_in=dim_token, dim_latent=dim_out, 
            beta=beta, 
            masked=(masking is not None and masking.lower() != 'none'), 
            skipgram=(masking=='skipgram'),
            mask_with=mask_with, 
            batch_first=batch_first, 
            padding_token=padding_token,
            mask_agg=mask_agg
        )
        super().__init__(
            dim_in, dim_out, dim_token, segmentation_name, max_context_size, model, padding_token, 
            masking, mask_with, mask_agg, beta, categorical, batch_first, corpus, prev_lvl, hyperparams, dirname,)

"""
def build_cnns(dim_out_prev, dim_out, max_context_size_prev, max_context_size, batch_first=False, n=3, n_fc=128):
    def build_single_enc(nin, nout):
        return nn.Sequential(nn.Conv1d(in_channels=nin, out_channels=nout, kernel_size=3, padding='same'), nn.ReLU())

    dim_after_convs = dim_out_prev*(2**n)*max_context_size_prev
    encoder = nn.Sequential(
        LambdaModule(lambda x: x.view(-1, max_context_size_prev, dim_out_prev)), # flatten the batch & current_context_size dims
        LambdaModule(lambda x: x.transpose(1, 2)), # swap the seq and token axis, now shape (batch, dim_out_prev, max_context_size)
        *[build_single_enc(dim_out_prev*(2**i), dim_out_prev*(2**(i+1))) for i in range(n)],  # now shape (batch, dim_out_prev*(2**n), max_context_size)
        LambdaModule(lambda x: x.view((-1, dim_after_convs))), # flatten the representations (batch, -1)
        nn.Sequential(nn.Linear(dim_after_convs, n_fc), nn.ReLU()), 
        nn.Sequential(nn.Linear(n_fc, n_fc), nn.ReLU()),
        nn.Linear(n_fc, dim_out), # (batch, dim_out)
        LambdaModule(lambda x: x.view((-1, max_context_size, dim_out)) if batch_first else x.view((max_context_size, -1, dim_out))),
    )

    def build_single_dec(nin, nout):
        return nn.Sequential(nn.ConvTranspose1d(in_channels=nin, out_channels=nout, kernel_size=3, padding=1), nn.ReLU())
    decoder = nn.Sequential(
        LambdaModule(lambda x: x.view((-1, dim_out))),
        nn.Sequential(nn.Linear(dim_out, n_fc), nn.ReLU()),
        nn.Sequential(nn.Linear(n_fc, n_fc), nn.ReLU()),
        nn.Linear(n_fc, dim_after_convs),
        LambdaModule(lambda x: x.view((-1, dim_out_prev*(2**n), max_context_size_prev))),
        *[build_single_dec(dim_out_prev*(2**(i+1)), dim_out_prev*(2**i)) for i in reversed(range(n))],
        LambdaModule(lambda x: x.transpose(1, 2).contiguous()), # swap seq and token axis again to have (-1, seq, token)
        LambdaModule(lambda x: x.view((-1, max_context_size, max_context_size_prev*dim_out_prev)) if batch_first else x.view((max_context_size, -1, max_context_size_prev*dim_out_prev)))
    )
    return encoder, decoder
    """
def build_cnns(dim_out_prev, dim_out, max_context_size_prev, max_context_size, batch_first=False, n=3, n_fc=128, n_conv_filters=64):
    dim_after_convs = dim_out_prev*(2**n)*max_context_size_prev
    encoder = nn.Sequential(
        LambdaModule(lambda x: x.view(-1, max_context_size_prev, dim_out_prev)), # flatten the batch & current_context_size dims
        LambdaModule(lambda x: x.transpose(1, 2)), # swap the seq and token axis, now shape (batch, dim_out_prev, max_context_size)
        nn.Sequential(nn.Conv1d(in_channels=dim_out_prev, out_channels=n_conv_filters, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2)),
        nn.Sequential(nn.Conv1d(in_channels=n_conv_filters, out_channels=n_conv_filters, kernel_size=3, padding=1), nn.ReLU()),
        nn.Sequential(nn.Conv1d(in_channels=n_conv_filters, out_channels=n_conv_filters, kernel_size=3, padding=1), nn.ReLU()),
        nn.Sequential(nn.Conv1d(in_channels=n_conv_filters, out_channels=n_conv_filters, kernel_size=3, padding=0), nn.ReLU(), nn.MaxPool1d(2)),
        #*[build_single_enc(dim_out_prev*(2**i), dim_out_prev*(2**(i+1))) for i in range(n)],  # now shape (batch, dim_out_prev*(2**n), max_context_size)
        LambdaModule(lambda x: x.view((-1, n_conv_filters*4))), # flatten the representations (batch, -1)
        nn.Sequential(nn.Linear(64*4, n_fc), nn.Dropout(.5)), 
        nn.Sequential(nn.Linear(n_fc, n_fc), nn.Dropout(.5)),
        nn.Linear(n_fc, dim_out), # (batch, dim_out)
        LambdaModule(lambda x: x.view((-1, max_context_size, dim_out)) if batch_first else x.view((max_context_size, -1, dim_out))),
    )

    decoder = nn.Sequential(
        LambdaModule(lambda x: x.view((-1, dim_out))),
        nn.Sequential(nn.Linear(dim_out, n_fc), nn.ReLU()),
        nn.Sequential(nn.Linear(n_fc, n_fc), nn.ReLU()),
        nn.Linear(n_fc, n_conv_filters*4),
        LambdaModule(lambda x: x.view((-1, n_conv_filters, 4))),
        nn.Sequential(nn.ConvTranspose1d(in_channels=n_conv_filters, out_channels=n_conv_filters, kernel_size=5, padding=0), nn.ReLU()),
        nn.Sequential(nn.ConvTranspose1d(in_channels=n_conv_filters, out_channels=n_conv_filters, kernel_size=5, padding=0), nn.ReLU()),
        nn.Sequential(nn.ConvTranspose1d(in_channels=n_conv_filters, out_channels=n_conv_filters, kernel_size=5, padding=0), nn.ReLU()),
        nn.Sequential(nn.ConvTranspose1d(in_channels=n_conv_filters, out_channels=dim_out_prev, kernel_size=5, padding=0), nn.ReLU()),
        #*[build_single_dec(dim_out_prev*(2**(i+1)), dim_out_prev*(2**i)) for i in reversed(range(n))],
        LambdaModule(lambda x: x.transpose(1, 2).contiguous()), # swap seq and token axis again to have (-1, seq, token)
        LambdaModule(lambda x: x.view((-1, max_context_size, max_context_size_prev*dim_out_prev)) if batch_first else x.view((max_context_size, -1, max_context_size_prev*dim_out_prev)))
    )
    return encoder, decoder

class CNNLevel(TrainableLevel):
    def __init__(
        self, 
        dim_out:int,
        segmentation_name: Key, max_context_size: int, 
        prev_lvl: AbstractLevel, 
        padding_token: Union[int, float] = 44, masking: str = None, mask_with: Union[int, float] = 0, mask_agg: str = 'max', beta: float = 0, 
        batch_first: bool = False, 
        corpus: StructuredCorpus = None, 
        hyperparams: dict = {}, 
        dirname: str = None,
        ):
        dim_in=prev_lvl.dim_out_concat
        dim_token=-1

        enc, dec = build_cnns(prev_lvl.dim_out, dim_out, prev_lvl.max_context_size, max_context_size, batch_first, **hyperparams)
        
        model=UniversalVAE(
            encoder=enc, 
            decoder=dec, 
            dim_encoder_out=dim_out, 
            dim_latent=dim_out, 
            categorical=False, 
            dim_decoder_in=dim_out, 
            beta=beta,
            masked=(masking is not None and masking.lower() != 'none'),
            skipgram=(masking=='skipgram'),
            mask_with=mask_with, 
            batch_first=batch_first, 
            padding_token=padding_token,
            mask_agg=mask_agg,
            )
        
        super().__init__(dim_in, dim_out, dim_token, segmentation_name, max_context_size, model, padding_token, masking, mask_with, mask_agg, beta, False, batch_first, corpus, prev_lvl, hyperparams, dirname)

    def load(dirname:str, corpus:StructuredCorpus=None, prev_lvl:AbstractLevel=None, run_name:str=None) -> TrainableLevel:
        with open(os.path.join(dirname, 'meta.yml'), 'r') as f: 
            file_content = yaml.load(f, Loader=SafeLoader)
        params = {}
        if 'level_params' in file_content:
            params = file_content['level_params']
        lvl = CNNLevel(
            dim_out=params['dim_out'], segmentation_name=params['segmentation_name'], 
            max_context_size=params['max_context_size'], prev_lvl=prev_lvl, 
            padding_token=params['padding_token'], masking=params['masking'], mask_with=params['mask_with'],
            mask_agg=params['mask_agg'], beta=params['beta'], batch_first=params['batch_first'], 
            corpus=corpus, hyperparams=params['hyperparams'], dirname=dirname)
        lvl.dirname = dirname
        if run_name is None:
            model_path = os.path.join(dirname, 'model.pt')
        else: 
            model_path = os.path.join(dirname, run_name, 'model.pt')
        if os.path.isfile(model_path):
            lvl.model.load_state_dict(torch.load(model_path))
        else: 
            print(f'Note: could not find model weights at {model_path}')
        return lvl

def get_max_segment_size(boundaries: List) -> int: 
    if len(boundaries) == 0:
        raise Exception("can't determine max. segment size from an empty list of boundaries")
    elif len(boundaries) == 1: 
        return boundaries[0] + 1
    else:             
        bnp = np.array(boundaries)
        diffs = bnp[1:] - bnp[:-1]
        return int(np.max(diffs))

"""
class Hierarchy(list):
    dirname: str

    def __init__(self, levels=[], dirname:str=None):
        super().__init__()
        for level in levels: 
            self.append(level)
        self.dirname = dirname

    def build(
        dim_in:int, 
        segmentation_per_level: Iterator[Key], 
        hyperparams:Union[dict, List[dict]], 
        dim_latent:Union[int, List[int]], 
        dim_token: Union[int, List[int]],
        padding_token: int,
        max_context_size: Union[int, List[int]]=auto_context_size, 
        decoder: Union[bool, List[bool]]=True,
        corpus: StructuredCorpus=None,
        mask_rate:float=0.0,
        mask_with:int=0,
        beta:float=1.0,
        batch_first: bool=False, 
        dirname: str=None,
        ) -> Hierarchy:

        if not type(hyperparams) == list: 
            hyperparams = [{**hyperparams} for _ in range(len(segmentation_per_level))]
        elif len(hyperparams) != len(segmentation_per_level): 
            raise Exception(f'length of argument list hyperparams ({len(hyperparams)}) must match length of segmentation_names ({len(segmentation_per_level)})')

        if not type(dim_latent) == list: 
            dim_latent = [dim_latent for _ in range(len(segmentation_per_level))]
        elif len(dim_latent) != len(segmentation_per_level): 
            raise Exception(f'length of argument list dim_latent ({len(dim_latent)}) must match length of segmentation_names ({len(segmentation_per_level)})')

        if not type(dim_token) == list: 
            dim_token = [dim_token for _ in range(len(segmentation_per_level))]
        elif len(dim_token) != len(segmentation_per_level): 
            raise Exception(f'length of argument list dim_token ({len(dim_token)}) must match length of segmentation_names ({len(segmentation_per_level)})')

        if not type(decoder) == list: 
            decoder = [decoder for _ in range(len(segmentation_per_level))]
        elif len(decoder) != len(segmentation_per_level): 
            raise Exception(f'length of argument list decoder ({len(decoder)}) must match length of segmentation_names ({len(segmentation_per_level)})')

        if not type(max_context_size) == list: 
            max_context_size = [max_context_size for _ in range(len(segmentation_per_level))]
        elif len(max_context_size) != len(segmentation_per_level): 
            raise Exception(f'length of argument list max_context_size ({len(max_context_size)}) must match length of segmentation_names ({len(segmentation_per_level)})')
        max_context_size = [auto_context_size if cs is None else cs for cs in max_context_size]

        if corpus is None: 
            for cs in max_context_size: 
                if cs is None: 
                    stringified = ','.join(map(str, max_context_size))
                    raise Exception(f'if no corpus is provided, all context sizes must be provided (got {stringified})')

        lvls = Hierarchy(dirname=dirname)
        if dirname is not None and not os.path.isdir(dirname):
            os.mkdir(dirname)

        zipped = zip(segmentation_per_level, hyperparams, dim_latent, max_context_size, dim_token, decoder)
        for i, (sname, _hp, _dl, _cs, _dt, dec) in enumerate(zipped):
            if len(lvls) == 0: 
                categorical=True
                prev_lvl=None
                dim_in=dim_in
            else:
                categorical=False
                prev_lvl=lvls[-1]
                dim_in=lvls.top_lvl().dim_out_concat

            lvl = Level(
                segmentation_name=sname,
                masked=mask_rate,
                mask_with=mask_with,
                padding_token=padding_token,
                decoder=dec,
                beta=beta,
                dim_in=dim_in,
                dim_out=_dl,
                dim_token=_dt,
                categorical=categorical,
                max_context_size=_cs,
                hyperparams=_hp,
                corpus=corpus,
                prev_lvl=prev_lvl,
                batch_first=batch_first,
                dirname=None if dirname is None else os.path.join(dirname, f'lvl{i}'),
            )
            lvls.append(lvl)

        if dirname is not None: 
            lvls.save()
        
        return lvls

    def save(self):
        if self.dirname is None: 
            raise Exception('must set a dirname before saving')
        if not os.path.isdir(self.dirname):
            os.mkdir(self.dirname)
        for i, lvl in enumerate(self):
            if lvl.dirname is None: 
                lvl.dirname = os.path.join(self.dirname, f'lvl{i}')
            lvl.save()

    def load(dirname:str, corpus: StructuredCorpus=None) -> Hierarchy:
        lvls = Hierarchy(dirname=dirname)
        for lvldir in sorted(os.listdir(dirname)):
            lvldir = os.path.join(dirname, lvldir)
            if os.path.isdir(lvldir): 
                lvl = Level.load(lvldir, corpus)
                if len(lvls) > 0:
                    lvl.prev_lvl = lvls.top_lvl()
                lvls.append(lvl)
        return lvls

    def __setitem__(self, key, val):
        raise Exception('use append method')

    def extend(self, other: list):
        raise Exception('no extending possible (use append)')

    def __getitem__(self, key: Union[slice, int]) -> Union[Level, Hierarchy]:
        if isinstance(key, slice):
            key = slice(0, key.stop, 1)
            return Hierarchy(super().__getitem__(key))

        elif isinstance(key, int):
            if key < 0: # Handle negative indices
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError(f"The index ({key}) is out of range.")
            return super().__getitem__(key) # Get the data from elsewhere
        else:
            raise TypeError("Invalid argument type.")

    def append(self, level: Level):
        if len(self) > 0:
            # check if dimension matches
            if level.dim_in != self[-1].dim_out_concat:
                raise LevelCompatibilityError(f"level dimension mismatch (level {len(self)} \
                    out {self.top_lvl().dim_out_concat}, new level in {level.dim_in})")

        # only 1st layer must be categorical
        if len(self) > 0:
            if level.categorical: 
                raise LevelCompatibilityError("non-root layers cannot be categorical")
        elif len(self) == 0:
            if not level.categorical: 
                raise LevelCompatibilityError("root level must be categorical")
        
        # combine keys so that segment boundaries match hierarchically
        for i in range(len(self)):
            prior_lvl = self[i]
            if not StructuredCorpus.keys_overlap(prior_lvl.segmentation_name, level.segmentation_name):
                prior_lvl.segmentation_name = StructuredCorpus.combine_key(prior_lvl.segmentation_name, level.segmentation_name)

        # set previous layer
        if len(self) > 0:
            level.prev_lvl = self.top_lvl()
        else: 
            level.prev_lvl = None

        super().append(level)
    
    def top_lvl(self) -> Level:
        return self[-1]

    def fit(self, training_params: dict, sample: bool=True, corpus: StructuredCorpus=None):
        return [lvl.fit_corpus(training_params, sample, corpus) for lvl in self]

    def predict(self, latent: Iterator) -> torch.Tensor:
        return self.top_lvl().predict(latent)

    def inverse_transform(self, latent: Iterator) -> StructuredCorpus:
        return self.top_lvl().inverse_transform(latent)
        """