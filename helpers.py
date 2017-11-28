from lxml import etree
import collections
import numpy as np
from random import shuffle
from shutil import copyfile

from bs4 import BeautifulSoup
from functools import reduce
import os

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix, hstack
import numpy as np

import datetime
from sklearn.neighbors import KNeighborsClassifier

def get_corpus():
    music = pickle.load(open( './data/music_corpus.p', 'rb' ))
    programming = pickle.load(open( './data/programming_corpus.p', 'rb' ))

    corpus = music + programming
    return corpus

def get_features(corpus):
    tokenize = lambda doc: doc.split(" ")

    sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
    sklearn_representation = sklearn_tfidf.fit_transform(corpus)
    return sklearn_representation

def get_sets(features):
    listofzeros = [0] * 1000
    listofones = [1] * 1000
    label = listofzeros + listofones

    matrix = hstack((features,np.array(label)[:,None])).A

    train = matrix[100:1900]
    np.random.shuffle(train)

    test_music = matrix[0:100]
    test_programming = matrix[1900:2000]

    test = np.concatenate((test_music, test_programming), axis=0)

    train_unlabeled = train[:,:-1]
    train_labels = train[:, -1]

    test_unlabeled = test[:,:-1]
    test_labels = test[:, -1]

    X_train, X_test, y_train, y_test = train_unlabeled, test_unlabeled, train_labels, test_labels

    return X_train, X_test, y_train, y_test

def train(name, clf, X_train, X_test, y_train, y_test):
    start = datetime.datetime.now()

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    acc = 100*score
    good = int(score*200)

    stop = datetime.datetime.now()
    length = int((stop-start).total_seconds())

    print("Using {} as a classifier, the accuracy was {}%, meaning that {} articles out of 200 were classified correctly. \
          \nFitting with this classifier took {} seconds".format(name, acc, good, length))

    return train, score
