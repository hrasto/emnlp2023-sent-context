from statistics import mode
from segmenters.structure import StructuredCorpus
from gensim.models import TfidfModel, LsiModel, LdaMulticore, LdaModel
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors

import module
import numpy as np
import os

dirs = [
    'swda_mfs_20w',
    #'swda_mfs_100w',
]
w2v_models=[
    '../models-gensim/glove-twitter-25',
    #'../models-gensim/glove-wiki-gigaword-100',
]

def gensim_corpus_to_numpy(corpus, num_topics):
    x = []
    for res in corpus:
        vec = np.zeros(num_topics)
        for i, val in res: vec[i] = val
        x.append(list(vec))
    x = np.array(x)
    return x

for dirname in dirs: 
    ctrain = StructuredCorpus.load(f'{dirname}/corpus/train')
    ctest = StructuredCorpus.load(f'{dirname}/corpus/test')
    cdev = StructuredCorpus.load(f'{dirname}/corpus/dev')
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

    if not os.path.isdir(f'{dirname}/sents_iso'):
        os.mkdir(f'{dirname}/sents_iso')

    for num_topics in [6, 12, 25]:
        print(num_topics, dirname)

        # LDA
        model_name = f'LDA_'+module.format_as_filename({'dim': num_topics})
        lsi = LsiModel(tfidf_train, num_topics=num_topics, id2word=dictionary)
        lsi_train = gensim_corpus_to_numpy(lsi[tfidf_train], num_topics)
        lsi_dev = gensim_corpus_to_numpy(lsi[tfidf_dev], num_topics)
        lsi_test = gensim_corpus_to_numpy(lsi[tfidf_test], num_topics)
        if not os.path.isdir(f'{dirname}/sents_iso/{model_name}'):
            os.mkdir(f'{dirname}/sents_iso/{model_name}')
        np.save(f'{dirname}/sents_iso/{model_name}/train.npy', lsi_train)
        np.save(f'{dirname}/sents_iso/{model_name}/dev.npy', lsi_dev)
        np.save(f'{dirname}/sents_iso/{model_name}/test.npy', lsi_test)

        # LSA
        model_name = f'LSA_'+module.format_as_filename({'dim': num_topics})
        lda = LdaModel(tfidf_train, num_topics=num_topics, id2word=dictionary)
        lda_train = gensim_corpus_to_numpy(lda[tfidf_train], num_topics)
        lda_dev = gensim_corpus_to_numpy(lda[tfidf_dev], num_topics)
        lda_test = gensim_corpus_to_numpy(lda[tfidf_test], num_topics)
        if not os.path.isdir(f'{dirname}/sents_iso/{model_name}'):
            os.mkdir(f'{dirname}/sents_iso/{model_name}')
        np.save(f'{dirname}/sents_iso/{model_name}/train.npy', lda_train)
        np.save(f'{dirname}/sents_iso/{model_name}/dev.npy', lda_dev)
        np.save(f'{dirname}/sents_iso/{model_name}/test.npy', lda_test)

    # GloVe 
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

    agg = 'mean'
    for w2v_path in w2v_models: 
        w2v = KeyedVectors.load(w2v_path)
        x_train = corpus2wv(ctrain, w2v, agg)
        x_dev = corpus2wv(cdev, w2v, agg)
        x_test = corpus2wv(ctest, w2v, agg)
        path_simple = w2v_path.split('/')[-1].upper()

        for num_topics in [6, 12, 25]:
            model_name = path_simple + '_' + module.format_as_filename({'dim': num_topics})

            if not os.path.isdir(f'{dirname}/sents_iso/{model_name}'):
                os.mkdir(f'{dirname}/sents_iso/{model_name}')

            np.save(f'{dirname}/sents_iso/{model_name}/train.npy', x_train[:, :num_topics])
            np.save(f'{dirname}/sents_iso/{model_name}/dev.npy', x_dev[:, :num_topics])
            np.save(f'{dirname}/sents_iso/{model_name}/test.npy', x_test[:, :num_topics])

    # Arbitrary word embeddings 
    def corpus2emb(corpus, word_vectors, dim, max_words=20):
        x = []
        for seq, labs in corpus[['default', 'act_tag']]:
            seq = seq[:max_words]
            _x = np.zeros((max_words, dim))
            _x[:len(seq)] = word_vectors[seq][:, :dim]
            _x = _x.mean(0)
            x.append(_x)
        x = np.array(x)
        return x 

    dim_token=8
    for num_topics in [6, 12, 25]:
        emb = module.init_embedding(nin=len(ctest), nout=num_topics)
        word_vectors = emb.weight.detach().numpy()
        x_train = corpus2emb(ctrain, word_vectors, num_topics)
        x_dev = corpus2emb(cdev, word_vectors, num_topics)
        x_test = corpus2emb(ctest, word_vectors, num_topics)
        assert x_train.shape[1] == num_topics and x_test.shape[1] == num_topics and x_dev.shape[1] == num_topics
        model_name = f'RAND{dim_token}_'+module.format_as_filename({'dim':num_topics})

        if not os.path.isdir(f'{dirname}/sents_iso/{model_name}'):
            os.mkdir(f'{dirname}/sents_iso/{model_name}')
            
        np.save(f'{dirname}/sents_iso/{model_name}/train.npy', x_train)
        np.save(f'{dirname}/sents_iso/{model_name}/dev.npy', x_dev)
        np.save(f'{dirname}/sents_iso/{model_name}/test.npy', x_test)