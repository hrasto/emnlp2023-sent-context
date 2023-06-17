from segmenters.structure import StructuredCorpus
from gensim.models import TfidfModel, LsiModel, LdaMulticore, LdaModel
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
import os

dirs = [
    'swda_mfs_20w',
    'swda_mfs_100w',
]
w2v_models=[
    '../models-gensim/glove-twitter-25',
    '../models-gensim/glove-wiki-gigaword-100',
]
num_topics=8

def gensim_corpus_to_numpy(corpus):
    x = []
    for res in corpus:
        vec = np.zeros(num_topics)
        for i, val in res: vec[i] = val
        x.append(list(vec))
    x = np.array(x)
    return x

for dirname in dirs: 
    print(dirname)
    ctrain = StructuredCorpus.load(f'{dirname}/corpus/train')
    ctest = StructuredCorpus.load(f'{dirname}/corpus/test')
    cdev = StructuredCorpus.load(f'{dirname}/corpus/dev')
    
    if not os.path.isdir(f'{dirname}/sents_iso'):
        os.mkdir(f'{dirname}/sents_iso')

    dictionary = Dictionary()
    dictionary.id2token = ctest.idx_to_word
    dictionary.token2id = ctest.word_to_idx

    bow_train = [dictionary.doc2bow(ctest.decode_sent(doc)) for doc in ctrain]
    tfidf = TfidfModel(bow_train)
    bow_dev = [dictionary.doc2bow(cdev.decode_sent(doc)) for doc in cdev]
    bow_test = [dictionary.doc2bow(ctest.decode_sent(doc)) for doc in ctest]
    tfidf_train = tfidf[bow_train]
    tfidf_dev = tfidf[bow_dev]
    tfidf_test = tfidf[bow_test]

    lsi = LsiModel(tfidf_train, num_topics=num_topics, id2word=dictionary)
    lsi_train = gensim_corpus_to_numpy(lsi[tfidf_train])
    lsi_dev = gensim_corpus_to_numpy(lsi[tfidf_dev])
    lsi_test = gensim_corpus_to_numpy(lsi[tfidf_test])
    if not os.path.isdir(f'{dirname}/sents_iso/lsi{num_topics}'):
        os.mkdir(f'{dirname}/sents_iso/lsi{num_topics}')
    np.save(f'{dirname}/sents_iso/lsi{num_topics}/train.npy', lsi_train)
    np.save(f'{dirname}/sents_iso/lsi{num_topics}/dev.npy', lsi_dev)
    np.save(f'{dirname}/sents_iso/lsi{num_topics}/test.npy', lsi_test)

    lda = LdaModel(tfidf_train, num_topics=num_topics, id2word=dictionary)
    lda_train = gensim_corpus_to_numpy(lda[tfidf_train])
    lda_dev = gensim_corpus_to_numpy(lda[tfidf_dev])
    lda_test = gensim_corpus_to_numpy(lda[tfidf_test])
    if not os.path.isdir(f'{dirname}/sents_iso/lda{num_topics}'):
        os.mkdir(f'{dirname}/sents_iso/lda{num_topics}')
    np.save(f'{dirname}/sents_iso/lda{num_topics}/train.npy', lda_train)
    np.save(f'{dirname}/sents_iso/lda{num_topics}/dev.npy', lda_dev)
    np.save(f'{dirname}/sents_iso/lda{num_topics}/test.npy', lda_test)

    def corpus2wv(corpus, w2v, agg):
        x = []
        for seq, labs in corpus[['default', 'act_tag']]:
            seq_w2v = [w2v.key_to_index.get(ctest.idx_to_word[i], 0) for i in seq]
            if agg == 'mean':
                x.append(w2v[seq_w2v].mean(0))
            elif agg=='max': 
                x.append(w2v[seq_w2v].max(0))
        x = np.array(x)
        return x 

    for agg in ['mean', 'max']: 
        for w2v_path in w2v_models: 
            w2v = KeyedVectors.load(w2v_path)
            x_train = corpus2wv(ctrain, w2v, agg)
            x_dev = corpus2wv(cdev, w2v, agg)
            x_test = corpus2wv(ctest, w2v, agg)

            path_simple = w2v_path.split('/')[-1]
            if not os.path.isdir(f'{dirname}/sents_iso/{path_simple}_{agg}'):
                os.mkdir(f'{dirname}/sents_iso/{path_simple}_{agg}')

            np.save(f'{dirname}/sents_iso/{path_simple}_{agg}/train.npy', x_train)
            np.save(f'{dirname}/sents_iso/{path_simple}_{agg}/dev.npy', x_dev)
            np.save(f'{dirname}/sents_iso/{path_simple}_{agg}/test.npy', x_test)