# -*- coding: utf-8 -*-


# RRN to classify text
# Author: adriamoya

# %matplotlib inline
#import matplotlib.pyplot as plt

import os
import re
import datetime
import numpy as np
import pandas as pd
from collections import Counter
import random as rn
import tensorflow as tf

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(1337)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see:
# https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)

from keras import backend as K
# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# Rest of code follows ...

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, LSTM, Embedding, Reshape, Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping

#import xgboost as xgb
#from xgboost import XGBClassifier

from sklearn import metrics
from sklearn.model_selection import train_test_split

# load data
df = pd.read_csv("../data/train.csv")
df_test = pd.read_csv("/../data/test.csv")

# ip calculation


def ip(y_target, y_pred):
    return 100 * (2 * (metrics.roc_auc_score(y_target, y_pred)) - 1)


def preprocessing(df, column="text"):
    """ Preprocessing (lower case, remove urls, punctuations) """

    print("\nPreprocessing %s ..." % (column))

    # preprocessing steps: lower case, remove urls, punctuations ...
    df[column] = df[column].str.lower()
    df[column] = df[column].str.replace(r'http[\w:/\.]+', '')  # remove urls
    # remove everything but characters and punctuation ( [^\.\w\s] )
    df[column] = df[column].str.replace(r'[^\.(a-zA-ZÀ-ÿ0-9)\s]', '')
    # remove dots in thousands (careful with decimals!)
    df[column] = df[column].str.replace(r'(?<=\d)(\.)(?=\d)', '')
    # replace multple periods with a single one
    df[column] = df[column].str.replace(r'\.\.+', '.')
    # replace multple periods with a single one
    df[column] = df[column].str.replace(r'\.', ' .')
    df[column] = df[column].str.replace(
        r'\(', ' ')  # replace brackets with white spaces
    # replace brackets with white spaces
    df[column] = df[column].str.replace(r'\)', ' ')
    # replace multple white space with a single one
    df[column] = df[column].str.replace(r'\s\s+', ' ')
    df[column] = df[column].str.strip()

    return df


def build_dictionary(df, min_count_word=5):
    """ Build dictionary and relationships between words and integers """

    print("\nBuilding dictionary ...")

    # get all unique words (only consider words that have been used more than
    # 5 times)
    all_text = ' '.join(df.text.values)
    words = all_text.split()
    u_words = Counter(words).most_common()
    # we will only consider words that have been used more than 5 times
    u_words = [word[0] for word in u_words if word[1] > min_count_word]

    print('The number of unique words is:', len(u_words))

    # create the dictionary
    word2num = dict(zip(u_words, range(len(u_words))))
    word2num['<Other>'] = len(u_words)
    num2word = dict(zip(word2num.values(), word2num.keys()))

    num2word[len(word2num)] = '<PAD>'
    word2num['<PAD>'] = len(word2num)

    return word2num, num2word, len(u_words)


def word2int(df, n_u_words, column='text', word_threshold=500):
    """ Convert words to integers and prepad sentences """

    print("\nConverting words to integers and prepadding ...")

    int_text = [[word2num[word] if word in word2num else n_u_words for word in Text.split(
    )] for Text in df[column].values]  # Text.split() python2

    print('The number of texts greater than %s in length is: ' % str(
        word_threshold), np.sum(np.array([len(t) > word_threshold for t in int_text])))
    print('The number of texts less than 50 in length is: ',
          np.sum(np.array([len(t) < 50 for t in int_text])))

    for i, t in enumerate(int_text):
        if len(t) < word_threshold:
            int_text[i] = [word2num['<PAD>']] * (word_threshold - len(t)) + t
        elif len(t) > word_threshold:
            int_text[i] = t[:word_threshold]
        else:
            continue

    return int_text


def fit_evaluate_model(X_train, X_valid, y_train, y_valid, params):
    """ Fit and evaluate Many to One RNN """

    print("\nCreating Sequential RNN: Many to One...")

    early_stopping = EarlyStopping(monitor='loss', patience=2)

    model = Sequential()

    # , batch_size=batch_size
    model.add(Embedding(len(word2num), params['embedding_size']))
    model.add(
        Conv1D(
            filters=128,
            kernel_size=5,
            padding='same',
            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(
        Conv1D(
            filters=128,
            kernel_size=5,
            padding='same',
            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    # model.add(Dropout(0.2))
    model.add(LSTM(100))
    # model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))  # sigmoid

    model.compile(
        loss=params['loss_func'],
        optimizer=params['optimizer'],
        metrics=params['metrics'])
    model.summary()

    batch_size = params['batch_size']
    print("\nFitting the model ...")
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=params['epochs'],
        callbacks=[early_stopping])

    print("\nPredicting probs on train ...")
    pred_train = model.predict(X_train)
    print("\nAUC: {0:.2f}%".format(100 * metrics.roc_auc_score(y_train,
                                                               pred_train)), "| GINI: {0:.2f}%".format(ip(y_train, pred_train)))

    print("\nEvaluating in valid ...")
    print(model.evaluate(X_valid, y_valid, batch_size=batch_size))

    print("\nPredicting probs on valid ...")
    pred_valid = model.predict(X_valid)
    print("\nAUC: {0:.2f}%".format(100 * metrics.roc_auc_score(y_valid,
                                                               pred_valid)), "| GINI: {0:.2f}%".format(ip(y_valid, pred_valid)))

    return model, pred_train, pred_valid


def predict_test(model, df_test, column):

    # words to numbers
    int_text = word2int(df_test, n_u_words, column, word_threshold)

    X = np.array(int_text)

    pred = model.predict(X)

    l_pred = []
    for item in pred:
        l_pred.append(item[0])

    return l_pred


# preprocessing steps: lower case, remove urls, punctuations ...


# text
df = preprocessing(df)
df_test = preprocessing(df_test)

# title
df = preprocessing(df, 'title')
df_test = preprocessing(df_test, 'title')

# build dictionary
min_count_word = 4
word2num, num2word, n_u_words = build_dictionary(df, min_count_word)

# train / valid split
print("\nTrain / Valid split ...")

np.random.seed(0)
df['msk'] = np.random.randn(df.shape[0])

np.random.seed(0)
msk = np.random.rand(len(df)) <= 0.9

df_train = df[msk]
df_train.reset_index(inplace=True)
df_train = df_train.drop(['msk', 'index'], axis=1)
df_valid = df[~msk]
df_valid.reset_index(inplace=True)
df_valid = df_valid.drop(['msk', 'index'], axis=1)

print("Train shape:", df_train.shape)
print("Valid shape:", df_valid.shape)

# ---------------------
# Model text
# ---------------------

word_threshold = 500

params = {
    'loss_func': 'binary_crossentropy',  # binary_crossentropy
    'optimizer': 'rmsprop',  # adam, rmsprop
    'metrics': ['accuracy'],
    'embedding_size': 100,
    'batch_size': 128,
    'epochs': 3
}

# word to integer
X_train = np.array(word2int(df_train, n_u_words, 'text', word_threshold))
X_valid = np.array(word2int(df_valid, n_u_words, 'text', word_threshold))

y_train = df_train['flag'].values
y_valid = df_valid['flag'].values

model_text, pred_train, pred_valid = fit_evaluate_model(
    X_train, X_valid, y_train, y_valid, params)

print("\nTest results ...")
test_pred = predict_test(model_text, df_test, 'text')

df_train['pred_text'] = pred_train
df_valid['pred_text'] = pred_valid
df_test['pred_text'] = test_pred

# ---------------------
# Model title
# ---------------------

word_threshold = 15

params = {
    'loss_func': 'binary_crossentropy',  # binary_crossentropy
    'optimizer': 'rmsprop',  # adam, rmsprop
    'metrics': ['accuracy'],
    'embedding_size': 100,
    'batch_size': 128,
    'epochs': 3
}

# word to integer
X_train = np.array(word2int(df_train, n_u_words, 'title', word_threshold))
X_valid = np.array(word2int(df_valid, n_u_words, 'title', word_threshold))

y_train = df_train['flag'].values
y_valid = df_valid['flag'].values

model_title, pred_train, pred_valid = fit_evaluate_model(
    X_train, X_valid, y_train, y_valid, params)

print("\nTest results ...")
test_pred = predict_test(model_title, df_test, 'title')

df_train['pred_title'] = pred_train
df_valid['pred_title'] = pred_valid
df_test['pred_title'] = test_pred

# ---------------------
# Stacking
# ---------------------

X_train = df_train[['pred_text', 'pred_title']].values
X_valid = df_valid[['pred_text', 'pred_title']].values

y_train = df_train['flag'].values
y_valid = df_valid['flag'].values


import xgboost as xgb
from xgboost import XGBClassifier

# xgb sparse matrix
xgtrain = xgb.DMatrix(X_train, label=y_train)
xgvalid = xgb.DMatrix(X_valid, label=y_valid)

clf = XGBClassifier(
    booster='gbtree',
    learning_rate=0.01,
    n_estimators=3000,  # 3000
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.7,
    colsample_bytree=0.7,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=99)

xgb_param = clf.get_xgb_params()

# cross-validation
# ------------------------------------------------------------------------------

cv_folds = 5
early_stopping_rounds = 100

print('\nInitializing cross-validation...')
cvresult = xgb.cv(
    xgb_param,
    xgtrain,
    num_boost_round=clf.get_params()['n_estimators'],
    nfold=cv_folds,
    metrics='auc',
    early_stopping_rounds=early_stopping_rounds,
    verbose_eval=1)

# retrieve parameters
print('\nXGBClassifier parameters')
clf.set_params(n_estimators=cvresult.shape[0])

# fit the algorithm on the training data
print('\nFit algorithm on train data...')
clf.fit(X_train, y_train, eval_metric='auc')

# Predict training set
# ------------------------------------------------------------------------------
print('\nPredicting on training set...')
dtrain_predictions = clf.predict(X_train)
dtrain_predprob = clf.predict_proba(X_train)[:, 1]

# print model report:
print('Model Report')
print('Accuracy : %.4g' % metrics.accuracy_score(y_train, dtrain_predictions))
print(
    'AUC Score (Train): %f' %
    metrics.roc_auc_score(
        y_train,
        dtrain_predprob))
print('IP Score  (Train): %f' % ip(y_train, dtrain_predprob))

# Predict valid set
# ------------------------------------------------------------------------------
print('\nPredicting on valid set...')
dvalid_predprob = clf.predict_proba(X_valid)[:, 1]

# print model report:
print('Model Report')
print(
    'AUC Score (Valid): %f' %
    metrics.roc_auc_score(
        y_valid,
        dvalid_predprob))
print('IP Score  (Valid): %f' % ip(y_valid, dvalid_predprob))

X_test = np.array(df_test[['pred_text', 'pred_title']])

# Predict test set
# ------------------------------------------------------------------------------
print('\nPredicting on test set...')
dtest_predprob = clf.predict_proba(X_test)[:, 1]

df_test['pred'] = dtest_predprob
