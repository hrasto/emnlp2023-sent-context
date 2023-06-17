import os
from segmenters.structure import StructuredCorpus
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score

dirs = [
    'swda_mfs_20w',
    'swda_mfs_100w',
]
n_labels=10
n_samples=10000
classifiers = {
    'lr': lambda: LogisticRegression(random_state=1, max_iter=2000),
    'svm_rbf': lambda: svm.SVC(kernel='rbf'),
    'tree5': lambda: tree.DecisionTreeClassifier(random_state=1, max_depth=5),
    'tree10': lambda: tree.DecisionTreeClassifier(random_state=1, max_depth=10),
    'tree20': lambda: tree.DecisionTreeClassifier(random_state=1, max_depth=20),
    'forest5': lambda: ensemble.RandomForestClassifier(random_state=1, max_depth=5),
    'forest10': lambda: ensemble.RandomForestClassifier(random_state=1, max_depth=10),
    'forest15': lambda: ensemble.RandomForestClassifier(random_state=1, max_depth=15),
    'knn10': lambda: KNeighborsClassifier(n_neighbors=10),
    'knn20': lambda: KNeighborsClassifier(n_neighbors=20),
    'knn30': lambda: KNeighborsClassifier(n_neighbors=30),
    'knn40': lambda: KNeighborsClassifier(n_neighbors=40),
}

for dirname in dirs: 
    print(dirname)
    ctrain = StructuredCorpus.load(f'{dirname}/corpus/train')
    ctest = StructuredCorpus.load(f'{dirname}/corpus/test')
    cdev = StructuredCorpus.load(f'{dirname}/corpus/dev')

    targets_train = [da[1][0] for _, da in ctrain[['default', 'act_tag']]]
    targets_dev = [da[1][0] for _, da in cdev[['default', 'act_tag']]]
    targets_test = [da[1][0] for _, da in ctest[['default', 'act_tag']]]
    
    targets_counts = dict(zip(*np.unique(targets_test, return_counts=True)))    
    targets_counts = sorted(targets_counts.items(), key=lambda x: x[1], reverse=True)
    def simplify_target(tg, most_common=n_labels-1, unk_label='?'):
        labels = [lab for lab, count in targets_counts[:most_common]]
        return tg[0] if tg[0] in labels else unk_label

    simple_targets_train = [simplify_target(t) for t in targets_train]
    simple_targets_dev = [simplify_target(t) for t in targets_dev]
    simple_targets_test = [simplify_target(t) for t in targets_test]
    
    simple_counts_train = list(zip(*np.unique(simple_targets_train, return_counts=True)))
    simple_counts_train = sorted(simple_counts_train, key=lambda x: x[1], reverse=True)
    simple_counts_dev = dict(zip(*np.unique(simple_targets_dev, return_counts=True)))
    simple_counts_test = dict(zip(*np.unique(simple_targets_test, return_counts=True)))
    freq_table = []
    for label, count in simple_counts_train: 
        freq_table.append({'label': label, 'train': count, 'dev': simple_counts_dev[label], 'test': simple_counts_test[label]})
    freq_table = pd.DataFrame(freq_table)
    freq_table['train%'] = (freq_table['train'] / freq_table['train'].sum())*100
    freq_table['dev%'] = (freq_table['dev'] / freq_table['dev'].sum())*100
    freq_table['test%'] = (freq_table['test'] / freq_table['test'].sum())*100
    freq_table = freq_table.round(2)
    print(freq_table)
    freq_table.to_csv(f'{dirname}/label_freq.csv', sep='\t')

    result_table = []
    sent_dirs = [name for name in os.listdir(f'{dirname}') if 'sents' in name]
    for rep_type in ['iso', 'con']: 
        if not os.path.isdir(f'{dirname}/sents_{rep_type}'): continue
        rep_names = os.listdir(f'{dirname}/sents_{rep_type}')
        for name in rep_names: 
            X_train = np.load(f'{dirname}/sents_{rep_type}/{name}/train.npy')
            X_test = np.load(f'{dirname}/sents_{rep_type}/{name}/test.npy')

            result = {
                'dataset': dirname, 
                'type': rep_type, 
                'representation': name, 
            }
            for clf_name, clf_init in classifiers.items():
                clf = clf_init()
                clf.fit(X_train[:n_samples], simple_targets_train[:n_samples])
                preds = clf.predict(X_test)
                acc = accuracy_score(simple_targets_test, preds)
                f1_micro = f1_score(simple_targets_test, preds, average='micro')
                f1_macro = f1_score(simple_targets_test, preds, average='macro')
                #print(f'{clf_name}\tacc={acc}\tf1_micro={f1_micro}\tf1_macro={f1_macro}')
                result[f'{clf_name}_acc'] = acc
                result[f'{clf_name}_f1ma'] = f1_macro
                result[f'{clf_name}_f1mi'] = f1_micro
            result_table.append(result)
    result_table = pd.DataFrame(result_table)
    result_table = result_table.round(2)
    result_table.to_csv(f'{dirname}/results.csv', sep='\t')