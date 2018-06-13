# -*- coding: utf-8 -*-
"""Modelling.

This module contains the training of the different algorithms to classify
articles.

"""
import re
import pickle
import warnings
import datetime
import numpy as np
import pandas as pd
import random as rn
from collections import Counter

from sklearn import metrics
from sklearn.model_selection import train_test_split

# import xgboost as xgb
# from xgboost import XGBClassifier

import tensorflow as tf

from keras.models import Sequential
from keras.layers import (Activation, Dropout, Flatten, Dense,
BatchNormalization, LSTM, Embedding, Reshape,
Conv1D, MaxPooling1D)
from keras.callbacks import EarlyStopping

from keras import backend as K
# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
tf.set_random_seed(1234)

pd.options.mode.chained_assignment = None
warnings.filterwarnings(module='sklearn*', action='ignore',
                        category=DeprecationWarning)

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(1337)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

#######################################################

# AUXILIAR FUNCTIONS
# ==================

def ip(y_target, y_pred):
    """ Gini calculation """
    return 100*(2*(metrics.roc_auc_score(y_target, y_pred))-1)


def preprocessing(df, column="text"):
    """ Preprocessing (lower case, remove urls, punctuations) """

    print("\nPreprocessing %s ..." % (column))

    df[column] = df[column].str.lower()
    df[column] = df[column].str.replace(r'http[\w:/\.]+', '')  # ( [^\.\w\s] )
    df[column] = df[column].str.replace(r'[^\.(a-zA-ZÀ-ÿ0-9)\s]', '')
    df[column] = df[column].str.replace(r'(?<=\d)(\.)(?=\d)', '')
    df[column] = df[column].str.replace(r'\.\.+', '.')
    df[column] = df[column].str.replace(r'\.', ' .')
    df[column] = df[column].str.replace(r'\(', ' ')
    df[column] = df[column].str.replace(r'\)', ' ')
    df[column] = df[column].str.replace(r'\s\s+', ' ')
    df[column] = df[column].str.strip()

    return df


def build_dictionary(df, min_count_word=5):
    """Build dictionary and relationships between words and integers.

    Args:
        df             : Dataset with articles information (pandas.DataFrame).
        min_count_word : Only consider words that have been used more
                         than n times. Default is 5.

    Returns:
        word2num       : Dictionary (words to numbers).
        num2word       : Dictionary (numbers to words).
        n_u_words      : Length of the dictionary (number of unique words).

    """

    print("\nBuilding dictionary ...")

    # Get all unique words (only consider frequent words)
    all_text = ' '.join(df.text.values)
    words = all_text.split()
    u_words = Counter(words).most_common()
    u_words = [word[0] for word in u_words if word[1] > min_count_word]

    print('The number of unique words is:', "{:,}".format(len(u_words)))

    # Create the dictionary
    word2num = dict(zip(u_words, range(len(u_words))))
    word2num['<Other>'] = len(u_words)
    num2word = dict(zip(word2num.values(), word2num.keys()))

    num2word[len(word2num)] = '<PAD>'
    word2num['<PAD>'] = len(word2num)

    n_u_words = len(u_words)

    return word2num, num2word, n_u_words


def word2int(df, n_u_words, column='text', word_threshold=500):
    """Convert words to integers and prepad sentences

    Args:
        df             : Dataset with articles information (pandas.DataFrame)
        n_u_words      : Length of the dictionary (number of unique words).
        column         : Name of the column that contains the
                         text of the article. Default is `text`.
        word_threshold : Number of words to consider for each text (padding).
                         Default is 500.

    Returns:
        int_text       : Array with texts translated to integers.

    """

    print("\nConverting words to integers and prepadding ...")

    int_text = [[word2num[word] if word in word2num else n_u_words
                 for word in Text.split()] for Text in df[column].values]

    print('The number of texts greater than %s in length is: ' %
          str(word_threshold),
          "{:,}".format(
              np.sum(np.array([len(t) > word_threshold for t in int_text]))))
    print('The number of texts less than 50 in length is: ',
          "{:,}".format(np.sum(np.array([len(t) < 50 for t in int_text]))))

    for i, t in enumerate(int_text):
        if len(t) < word_threshold:
            int_text[i] = [word2num['<PAD>']]*(word_threshold-len(t)) + t
        elif len(t) > word_threshold:
            int_text[i] = t[:word_threshold]
        else:
            continue

    return int_text


def fit_evaluate_model(X_train, X_valid, y_train, y_valid, params):
    """ Fit and evaluate Many to One RNN

    Args:
        X_train    : Array with train features.
        X_valid    : Array with validation features.
        y_train    : Array with train flag.
        y_valid    : Array with validation flag.
        params     : Dictionary with parameter configuration.

    Returns:
        model      : Model already trained.
        pred_train : Array with train predictions.
        pred_valid : Array with validation predictions.

    """

    print("\nCreating Sequential RNN: Many to One...")

    early_stopping = EarlyStopping(monitor='loss', patience=2)

    model = Sequential()

    model.add(Embedding(len(word2num), params['embedding_size']))
    model.add(Conv1D(filters=128, kernel_size=5, padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=5, padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))  # sigmoid

    model.compile(loss=params['loss_func'], optimizer=params['optimizer'],
                  metrics=params['metrics'])
    model.summary()

    batch_size = params['batch_size']
    print("\nFitting the model ...")
    model.fit(X_train, y_train, batch_size=batch_size, epochs=params['epochs'],
              callbacks=[early_stopping])

    print("\nPredicting probs on train ...")
    pred_train = model.predict(X_train)
    print("AUC: {0:.2f}%".format(100*metrics.roc_auc_score(y_train,
                                                             pred_train)),
          "| GINI: {0:.2f}%".format(ip(y_train, pred_train)))

    # print("\nEvaluating in valid ...")
    # print(model.evaluate(X_valid, y_valid, batch_size=batch_size))

    print("\nPredicting probs on valid ...")
    pred_valid = model.predict(X_valid)
    print("AUC: {0:.2f}%".format(100*metrics.roc_auc_score(y_valid,
                                                             pred_valid)),
          "| GINI: {0:.2f}%".format(ip(y_valid, pred_valid)))

    return model, pred_train, pred_valid



def predict_test(model, column,  X_test, y_test=None):
    """Make predictions in test dataset.

    Args:
        model     : Model trained.
        X_test    : Array with test feature (already word2int).

    Returns:
        pred_test : Array with test predictions.

    """

    pred = model.predict(X_test)

    pred_test = []
    for item in pred:
        pred_test.append(item[0])

    try:
        print("\nPredicting probs on test ...")
        print("AUC: {0:.2f}%".format(100*metrics.roc_auc_score(y_test,
                                                                 pred_test)),
              "| GINI: {0:.2f}%".format(ip(y_test, pred_test)))
    except:
        pass

    return pred_test

#######################################################

flag = 'flag'

# Load data
df = pd.read_csv("../data/train.csv")
df_test = pd.read_csv("../data/test_solution.csv")

# PREPROCESSING
# =============

# Text processing
# ---------------
df = preprocessing(df, 'text')
df_test = preprocessing(df_test, 'text')

# Title processing
# ----------------
df = preprocessing(df, 'title')
df_test = preprocessing(df_test, 'title')

# Building dictionary
# -------------------
min_count_word = 4
word2num, num2word, n_u_words = build_dictionary(df, min_count_word)

# Train / Validation split
# ------------------------
print("\nTrain / Valid split ...")

X, y = df[df.columns[~df.columns.str.contains(flag)]].values, df[flag].values
X_train, X_valid, y_train, y_valid = train_test_split(X,
                                                      y,
                                                      test_size=0.2,
                                                      stratify=y,
                                                      shuffle=True,
                                                      random_state=42)

y_test = df_test[flag].values

print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)
print("y_train:", y_train.shape)
print("y_valid:", y_valid.shape)

columns = df.columns[~df.columns.str.contains(flag)].values

df_train = pd.DataFrame(X_train, columns=columns); df_train[flag] = y_train
df_valid = pd.DataFrame(X_valid, columns=columns); df_valid[flag] = y_valid

print("\nTrain  :", df_train.shape)
print("Valid  :", df_valid.shape)
print("Test   :", df_test.shape)


# MODELLING
# =========

# Text
# ----

# word_threshold = 500
#
# params = {
#     'loss_func': 'binary_crossentropy',  # binary_crossentropy
#     'optimizer': 'rmsprop',  # adam, rmsprop
#     'metrics': ['accuracy'],
#     'embedding_size': 100,
#     'batch_size': 128,
#     'epochs': 3
# }
#
# X_train = np.array(word2int(df_train, n_u_words, 'text', word_threshold))
# X_valid = np.array(word2int(df_valid, n_u_words, 'text', word_threshold))
#
# y_train = df_train['flag'].values
# y_valid = df_valid['flag'].values
#
# model_text, pred_train, pred_valid = fit_evaluate_model(X_train, X_valid,
#                                                         y_train, y_valid,
#                                                         params)
#
# print("\nTest results ...")
# pred_test = predict_test(model_text, 'text', X_test, y_test)
#
# df_train['pred_text'] = pred_train
# df_valid['pred_text'] = pred_valid
# df_test['pred_text'] = pred_test

# Title
# -----

word_threshold = 15

params = {
    'loss_func': 'binary_crossentropy',  # binary_crossentropy
    'optimizer': 'rmsprop',  # adam, rmsprop
    'metrics': ['accuracy'],
    'embedding_size': 100,
    'batch_size': 128,
    'epochs': 3
}

X_train = np.array(word2int(df_train, n_u_words, 'title', word_threshold))
X_valid = np.array(word2int(df_valid, n_u_words, 'title', word_threshold))
X_test = np.array(word2int(df_test, n_u_words, 'title', word_threshold))

y_train = df_train['flag'].values
y_valid = df_valid['flag'].values

model_title, pred_train, pred_valid = fit_evaluate_model(X_train, X_valid,
                                                         y_train, y_valid,
                                                         params)

pred_test = predict_test(model_title, 'title', X_test, y_test)

df_train['pred_title'] = pred_train
df_valid['pred_title'] = pred_valid
df_test['pred_title'] = pred_test

# Summary
# -------

word_threshold = 250

params = {
    'loss_func': 'binary_crossentropy',  # binary_crossentropy
    'optimizer': 'rmsprop',  # adam, rmsprop
    'metrics': ['accuracy'],
    'embedding_size': 100,
    'batch_size': 128,
    'epochs': 3
}

X_train = np.array(word2int(df_train, n_u_words, 'summary', word_threshold))
X_valid = np.array(word2int(df_valid, n_u_words, 'summary', word_threshold))
X_test = np.array(word2int(df_test, n_u_words, 'summary', word_threshold))

y_train = df_train['flag'].values
y_valid = df_valid['flag'].values

model_summary, pred_train, pred_valid = fit_evaluate_model(X_train, X_valid,
                                                           y_train, y_valid,
                                                           params)

pred_test = predict_test(model_summary, 'summary', X_test, y_test)

df_train['pred_summary'] = pred_train
df_valid['pred_summary'] = pred_valid
df_test['pred_summary'] = pred_test

# Stacking
# --------

# def fit_evaluate_xgboost(alg,
#                          dtrain,
#                          dtest,
#                          predictors,
#                          verbose=0,
#                          useTrainCV=True,
#                          cv_folds=5,
#                          early_stopping_rounds=50,
#                          flag='flag'):
#
#     if useTrainCV:
#         xgb_param = alg.get_xgb_params()
#         xgtrain = xgb.DMatrix(dtrain[predictors].values,
#                               label=dtrain[flag].values.flatten())
#         cvresult = xgb.cv(xgb_param,
#                           xgtrain,
#                           num_boost_round=alg.get_params()['n_estimators'],
#                           nfold=cv_folds,
#                           metrics='auc',
#                           early_stopping_rounds=early_stopping_rounds,
#                           verbose_eval=verbose)
#         alg.set_params(n_estimators=cvresult.shape[0])
#         print(alg.get_params())
#
#     # Fit the algorithm on the data
#     alg.fit(dtrain[predictors], dtrain[flag].values.flatten(),
#             eval_metric='auc')
#
#     # Predict training set:
#     dtrain_predictions = alg.predict(dtrain[predictors])
#     dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]
#
#     # Print model report:
#     print("\nModel Report (Train)")
#     print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[flag].values,
#                                                      dtrain_predictions))
#     print("AUC Score: %f" % metrics.roc_auc_score(dtrain[flag].values,
#                                                   dtrain_predprob))
#
#     # Predict validation set:
#     dtest_predprob = alg.predict_proba(dtest[predictors])[:, 1]
#
#     # Print model report:
#     print("\nModel Report (Test)")
#     print("AUC Score: %f" % metrics.roc_auc_score(dtest[flag].values,
#                                                   dtest_predprob))
#
#     return alg
#
# predictors = ['pred_text', 'pred_title', 'pred_summary']
#
# X_train = df_train[[predictors]].values
# X_valid = df_valid[[predictors]].values
#
# # xgb sparse matrix
#
# clf = XGBClassifier(
#     booster='gbtree',
#     learning_rate=0.01,
#     n_estimators=3000,  # 3000
#     max_depth=5,
#     min_child_weight=1,
#     gamma=0,
#     subsample=0.7,
#     colsample_bytree=0.7,
#     objective='binary:logistic',
#     nthread=4,
#     scale_pos_weight=1,
#     seed=99)
#
# model_xgb = fit_evaluate_xgboost(model_xgb, df_train, df_valid, predictors)
