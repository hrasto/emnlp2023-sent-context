import os
from segmenters.structure import StructuredCorpus

dirs = [
    'swda_mfs_20w',
    'swda_mfs_100w',
]

for dirname in dirs: 
    print(dirname)
    ctrain = StructuredCorpus.load(f'{dirname}/corpus/train')
    ctest = StructuredCorpus.load(f'{dirname}/corpus/test')
    cdev = StructuredCorpus.load(f'{dirname}/corpus/dev')

    if not os.path.isdir(f'{dirname}/sents_con'):
        os.mkdir(f'{dirname}/sents_con')
        