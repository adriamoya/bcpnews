# -*- coding: utf-8 -*-
"""Modelling.

This module contains the training of the different algorithms to classify
articles.

"""
from numpy.random import seed
seed(1)

from tensorflow import set_random_seed
set_random_seed(2)

import os
import re
import pickle
import datetime
import numpy as np
import pandas as pd
from collections import Counter
import random as rn
import tensorflow as tf


# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K
# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
# rn.seed(12345)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# Rest of code follows ...
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.layers import (Activation, Bidirectional, Dropout, TimeDistributed,
                          Flatten, Dense, BatchNormalization, LSTM, Embedding,
                          Reshape, Conv1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D)

from keras.callbacks import EarlyStopping

import xgboost as xgb
from xgboost import XGBClassifier

import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV

#######################################################

# AUXILIAR FUNCTIONS
# ==================

def prepare_text_data(train_input, test_input, max_words, max_len):
    """ Prepare text data """
    print('\nTokenizing and padding data... (max_words: %d)' % max_words)
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(train_input)
    sequences_train = tok.texts_to_sequences(train_input)
    sequences_test = tok.texts_to_sequences(test_input)

    print('Pad sequences (max_len: %d)' % max_len)
    train_input_f = sequence.pad_sequences(sequences_train, maxlen=max_len)
    test_input_f = sequence.pad_sequences(sequences_test, maxlen=max_len)

    return train_input_f, test_input_f


def model_CNN(X_tr, X_te, y_tr, y_te, te_index, oof_train, oof_valid_skf, params):
    
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1)

    model = Sequential()
    model.add(Embedding(params['max_features'], params['embedding_size'], input_length=params['max_len']))
    model.add(Dropout(0.2))
    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters=params['filters'], kernel_size=params['kernel_size'], padding='valid', activation='relu', strides=1))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())
    # We add a vanilla hidden layer:
    model.add(Dense(params['hidden_dims']))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=params['loss_func'],
                  optimizer=params['optimizer'],
                  metrics=params['metrics'])
    
    if fold_counter == 0:
        model.summary()
        
    model.fit(X_tr, 
              y_tr, 
              batch_size=params['batch_size'],
              epochs=params['epochs'],
              validation_data=(X_te, y_te),
              callbacks=[early_stopping],
              verbose=0)

    oof_train[te_index] = model.predict(X_te)[:, 0]
    oof_valid_skf[params['fold_counter'], :] = model.predict(X_valid)[:, 0]
    score = 100*roc_auc_score(y_valid, oof_valid_skf[params['fold_counter'], :])
    print('fold %d: [%.4f]' % (params['fold_counter']+1, score))  
    
    model.save('models/CNN_%s_%s.h5' % (params['feature'], params['fold_counter']))

    return oof_train, oof_valid_skf


def model_LSTM(X_tr, X_te, y_tr, y_te, te_index, oof_train, oof_valid_skf, params):
    
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1)
    
    model = Sequential()
    model.add(Embedding(params['max_features'], params['embedding_size'], input_length=params['max_len']))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=params['loss_func'],
                  optimizer=params['optimizer'],
                  metrics=params['metrics'])
    
    if fold_counter == 0:
        model.summary()
        
    model.fit(X_tr, 
              y_tr, 
              batch_size=params['batch_size'],
              epochs=params['epochs'],
              validation_data=(X_te, y_te),
              callbacks=[early_stopping],
              verbose=0)
    
    oof_train[te_index] = model.predict(X_te)[:, 0]
    oof_valid_skf[params['fold_counter'], :] = model.predict(X_valid)[:, 0]
    score = 100*roc_auc_score(y_valid, oof_valid_skf[params['fold_counter'], :])
    print('fold %d: [%.4f]' % (params['fold_counter']+1, score))  
    
    model.save('models/LSTM_%s_%s.h5' % (params['feature'], params['fold_counter']))

    return oof_train, oof_valid_skf


def model_BiLSTM(X_tr, X_te, y_tr, y_te, te_index, oof_train, oof_valid_skf, params):
    
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1)
    
    model = Sequential()
    model.add(Embedding(params['max_features'], params['embedding_size'], input_length=params['max_len']))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=params['loss_func'],
                  optimizer=params['optimizer'],
                  metrics=params['metrics'])
    
    if fold_counter == 0:
        model.summary()
        
    model.fit(X_tr, 
              y_tr, 
              batch_size=params['batch_size'],
              epochs=params['epochs'],
              validation_data=(X_te, y_te),
              callbacks=[early_stopping],
              verbose=0)
    
    oof_train[te_index] = model.predict(X_te)[:, 0]
    oof_valid_skf[params['fold_counter'], :] = model.predict(X_valid)[:, 0]
    score = 100*roc_auc_score(y_valid, oof_valid_skf[params['fold_counter'], :])
    print('fold %d: [%.4f]' % (params['fold_counter']+1, score))
    
    model.save('models/BiLSTM_%s_%s.h5' % (params['feature'], params['fold_counter']))

    return oof_train, oof_valid_skf


def model_CNNLSTM(X_tr, X_te, y_tr, y_te, te_index, oof_train, oof_valid_skf, params):
    
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1)
    
    model = Sequential()
    model.add(Embedding(params['max_features'], params['embedding_size'], input_length=params['max_len']))
    model.add(Conv1D(filters=params['filters'], kernel_size=params['kernel_size'], padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=params['pool_size']))
    model.add(Conv1D(filters=params['filters'], kernel_size=params['kernel_size'], padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=params['pool_size']))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid')) # sigmoid
    
    model.compile(loss=params['loss_func'],
                  optimizer=params['optimizer'],
                  metrics=params['metrics'])
    
    if fold_counter == 0:
        model.summary()

    model.fit(X_tr, 
              y_tr, 
              batch_size=params['batch_size'],
              epochs=params['epochs'],
              validation_data=(X_te, y_te),
              callbacks=[early_stopping],
              verbose=0)
    
    oof_train[te_index] = model.predict(X_te)[:, 0]
    oof_valid_skf[params['fold_counter'], :] = model.predict(X_valid)[:, 0]
    score = 100*roc_auc_score(y_valid, oof_valid_skf[params['fold_counter'], :])
    print('fold %d: [%.4f]' % (params['fold_counter']+1, score))
    
    model.save('models/CNNLSTM_%s_%s.h5' % (params['feature'], params['fold_counter']))

    return oof_train, oof_valid_skf

#######################################################

FLAG = 'flag'

# Load data
df = pd.read_csv("../../1_construction/3_newspaper_scraper/analyses/cleaned_datasets/train.csv")
df_test = pd.read_csv("../../1_construction/3_newspaper_scraper/analyses/cleaned_datasets/test.csv")

# Train / validation split
print("\nTrain / Validation split ...")

X, y = df[df.columns[~df.columns.str.contains(FLAG)]].values, df[FLAG].values
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, stratify=y, shuffle=True, random_state=42)

print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)
print("y_train:", y_train.shape)
print("y_valid:", y_valid.shape)

# X_test = df_test.values
# y_test = df_test[FLAG].values

columns = df.columns[~df.columns.str.contains(FLAG)].values

df_train = pd.DataFrame(X_train, columns=columns); df_train[FLAG] = y_train
df_valid = pd.DataFrame(X_valid, columns=columns); df_valid[FLAG] = y_valid

print("\nTrain  :", df_train.shape)
print("Valid  :", df_valid.shape)
print("Test   :", df_test.shape)


# MODELLING
# =========

# 1st level models
# -----------------------------------------------------
"""
NFOLDS = 4

features = [{"name": "title", "max_len": 20, "max_features": 10000},
            {"name": "summary", "max_len": 250, "max_features": 40000},
            {"name": "text", "max_len": 500, "max_features": 40000}]  # , {"feature": "summary", "word_threshold": 250}

models = ['CNN', 'LSTM', 'BiLSTM', 'CNNLSTM']

params = {
    'loss_func': 'binary_crossentropy', # binary_crossentropy
    'optimizer': 'adam', # adam, rmsprop
    'metrics': ['accuracy'],
    'embedding_size': 100,
    'batch_size': 128,
    'epochs': 3,
    'filters': 250, # 128
    'kernel_size': 3,
    'hidden_dims': 250,
    'pool_size': 2
}

train_level_2 = np.zeros((df_train.shape[0], len(models) * len(features)))
valid_level_2 = np.zeros((df_valid.shape[0], len(models) * len(features)))

for feature_counter, feature in enumerate(features):
    
    params['max_len'] = feature['max_len']
    params['max_features'] = feature['max_features']
    params['feature'] = feature['name']

    # word to integer
    X_train, X_valid =  prepare_text_data(df_train[feature['name']].values,
                                          df_valid[feature['name']].values,
                                          params['max_features'],
                                          params['max_len'])

    ntrain = X_train.shape[0]
    nvalid = X_valid.shape[0]
    
    for model_counter, model in enumerate(models):
        
        idx = feature_counter*len(models) + model_counter + 1

        print("")
        print("-"*80)
        print("Model {} of {}. {} architecture using feature '{}'".format(idx, len(models)*len(features), model, feature['name']))
        print("-"*80)

        oof_train = np.zeros((ntrain,))
        oof_valid = np.zeros((nvalid,))
        oof_valid_skf = np.empty((NFOLDS, nvalid))

        kf = StratifiedKFold(n_splits=NFOLDS, shuffle=False, random_state=0)

        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            
            params['fold_counter'] = fold_counter

            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]

            if model == "CNN":
                oof_train, oof_valid_skf = model_CNN(X_tr, X_te, y_tr, y_te, te_index, oof_train, oof_valid_skf, params)
            elif model == "LSTM":
                oof_train, oof_valid_skf = model_LSTM(X_tr, X_te, y_tr, y_te, te_index, oof_train, oof_valid_skf, params)
            elif model == "BiLSTM":
                oof_train, oof_valid_skf = model_BiLSTM(X_tr, X_te, y_tr, y_te, te_index, oof_train, oof_valid_skf, params)
            elif model == "CNNLSTM":
                oof_train, oof_valid_skf = model_CNNLSTM(X_tr, X_te, y_tr, y_te, te_index, oof_train, oof_valid_skf, params)                

        train_level_2[:, idx-1] = oof_train[:]
        
        print("\nAveraging scores in out of fold valid dataset...")
        oof_valid[:] = oof_valid_skf.mean(axis=0)
        valid_level_2[:, idx-1] = oof_valid[:]
        score = 100*metrics.roc_auc_score(y_valid, oof_valid[:])
        print('valid:  [%.4f]' % score)

"""
NFOLDS = 4

features = [{"name": "title", "max_len": 20, "max_features": 10000},
            {"name": "summary", "max_len": 250, "max_features": 40000},
            {"name": "text", "max_len": 500, "max_features": 40000}]  # , {"feature": "summary", "word_threshold": 250}
models = ['CNN', 'LSTM', 'BiLSTM', 'CNNLSTM']

params = {
    'loss_func': 'binary_crossentropy', # binary_crossentropy
    'optimizer': 'adam', # adam, rmsprop
    'metrics': ['accuracy'],
    'embedding_size': 100,
    'batch_size': 128,
    'epochs': 3,
    'filters': 250, # 128
    'kernel_size': 3,
    'hidden_dims': 250,
    'pool_size': 2
}

train_level_2 = np.zeros((df_train.shape[0], len(models) * len(features)))
valid_level_2 = np.zeros((df_valid.shape[0], len(models) * len(features)))
# test_level_2 = np.zeros((df_test.shape[0], len(models) * len(features)))

for feature_counter, feature in enumerate(features):
    
    params['max_len'] = feature['max_len']
    params['max_features'] = feature['max_features']
    params['feature'] = feature['name']

    # word to integer
    X_train, X_valid =  prepare_text_data(df_train[feature['name']].values,
                                          df_valid[feature['name']].values,
                                          params['max_features'],
                                          params['max_len'])

    # X_train, X_test = prepare_text_data(df_train[feature['name']].values,
    #                                     df_test[feature['name']].values,
    #                                     params['max_features'],
    #                                     params['max_len'])

    ntrain = X_train.shape[0]
    nvalid = X_valid.shape[0]
    # ntest = X_test.shape[0]
    
    for model_counter, model_name in enumerate(models):
        
        idx = feature_counter*len(models) + model_counter + 1

        print("")
        print("-"*80)
        print("Model {} of {}. {} architecture using feature '{}'".format(idx, len(models)*len(features), model_name, feature['name']))
        print("-"*80)

        oof_train = np.zeros((ntrain,))
        oof_valid = np.zeros((nvalid,))
        # oof_test = np.zeros((ntest,))

        oof_valid_skf = np.empty((NFOLDS, nvalid))
        # oof_test_skf = np.empty((NFOLDS, ntest))

        kf = StratifiedKFold(n_splits=NFOLDS, shuffle=False, random_state=0)

        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            
            params['fold_counter'] = fold_counter

            # Split data and target
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]

            model = load_model('models/%s_%s_%s.h5' % (model_name, feature['name'], fold_counter))

            oof_train[te_index] = model.predict(X_te)[:, 0]
            oof_valid_skf[params['fold_counter'], :] = model.predict(X_valid)[:, 0]
            # oof_test_skf[params['fold_counter'], :] = model.predict(X_test)[:, 0]

            score = 100*roc_auc_score(y_valid, oof_valid_skf[params['fold_counter'], :])
            print('fold %d: [%.4f]' % (params['fold_counter']+1, score))         

        train_level_2[:, idx-1] = oof_train[:]
        
        print("\nAveraging scores in out of fold valid dataset...")
        oof_valid[:] = oof_valid_skf.mean(axis=0)
        valid_level_2[:, idx-1] = oof_valid[:]
        score = 100*roc_auc_score(y_valid, oof_valid[:])
        print('valid:  [%.4f]' % score)

        # print("\nAveraging scores in out of fold test dataset...")
        # oof_test[:] = oof_test_skf.mean(axis=0)
        # test_level_2[:, idx-1] = oof_test[:]


# 2nd level model
# -----------------------------------------------------
"""
params = {
        'min_child_weight': [1, 3, 5],
        'gamma': [0.5, 1, 1.5, 2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.1, 0.01, 0.005]
        }

xgb = XGBClassifier(learning_rate=0.001, n_estimators=10000,
                    objective='binary:logistic', silent=True)
PARAM_COMB = 5

skf = StratifiedKFold(n_splits=NFOLDS, shuffle = True, random_state = 1001)

print("Randomized search...")
random_search = RandomizedSearchCV(xgb,
                                   param_distributions=params,
                                   n_iter=PARAM_COMB,
                                   scoring='roc_auc',
                                   n_jobs=-1,
                                   cv=skf.split(train_level_2, y_train),
                                   verbose=1,  # 2
                                   random_state=1001 )

random_search.fit(train_level_2, y_train)

pickle.dump(random_search.best_estimator_, open('models/xgboost_level_2.dat', "wb"))

pred_train = random_search.predict_proba(train_level_2)[:, 1]
cv_train_auc = roc_auc_score(y_train, pred_train)
print('CV train with XGBoost AUC: {}'.format(cv_train_auc))

pred_valid = random_search.predict_proba(valid_level_2)[:, 1]
cv_valid_auc = roc_auc_score(y_valid, pred_valid)
print('CV valid with XGBoost AUC: {}'.format(cv_valid_auc))

"""
random_search = pickle.load(open('models/xgboost_level_2.dat', "rb"))

pred_train = random_search.predict_proba(train_level_2)[:, 1]
cv_train_auc = roc_auc_score(y_train, pred_train)
print('CV train with XGBoost AUC: [%.4f]' % cv_train_auc)

pred_valid = random_search.predict_proba(valid_level_2)[:, 1]
cv_valid_auc = roc_auc_score(y_valid, pred_valid)
print('CV valid with XGBoost AUC: [%.4f]' % cv_valid_auc)

# pred_test = random_search.predict_proba(test_level_2)[:, 1]
