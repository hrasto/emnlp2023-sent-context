import os
from segmenters.structure import StructuredCorpus
import torch
import torch.nn as nn
from torch import Tensor as T
import torch.nn.functional as F
from module import ReconstructionModel, TokenProjector

dirs = [
    'swda_mfs_20w',
    'swda_mfs_100w',
]

tp = TokenProjector(8)
u = torch.rand((16, 8))
v = torch.rand((16, 8))
y = torch.randint(0, 2, (16,))
yh, y = tp(u, v, y)
tp.loss_fn(yh, y)

def generate_batch(bs):
    

exit()

for dirname in dirs: 
    print(dirname)
    #ctrain = StructuredCorpus.load(f'{dirname}/corpus/train')
    ctest = StructuredCorpus.load(f'{dirname}/corpus/test')
    cdev = StructuredCorpus.load(f'{dirname}/corpus/dev')

    if not os.path.isdir(f'{dirname}/sents_con'):
        os.mkdir(f'{dirname}/sents_con')