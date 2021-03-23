import json
import pandas as pd
import h5py
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Dropout, LSTM, Flatten, GRU,TimeDistributed, Conv1D
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
import os
import h5py
import matplotlib.pyplot as plt
from keras import regularizers
from tensorflow.keras.regularizers import l1
import ast
from tqdm import tqdm
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, truncnorm, randint
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.stats import loguniform
from pandas import read_csv
from keras.wrappers.scikit_learn import KerasClassifier
features = np.load('lfeatures.npy', allow_pickle=True)
labels = np.load('llabels.npy')
X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size = 0.1, random_state = 42)

dropout = 0.5
units = 512
dense1units = 256
dense2units = 128
dense3units = 64
def getModel(units, dropout, optimizer, dense1units, dense2units, dense3units):
    model = Sequential()
    model.add(LSTM(512, kernel_initializer = 'VarianceScaling', name = 'lstm1'))
    model.add(Dense(256, activation='relu', kernel_initializer='lecun_uniform', name='fc5'))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(128, activation='relu', kernel_initializer='lecun_uniform', name='fc5'))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(64, activation='relu', kernel_initializer='lecun_uniform', name='fc5'))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'output_sigmoid'))
    model.compile(optimizer= optimizer , loss = tf.losses.categorical_crossentropy , metrics=['accuracy'])
    return model
    

optimizer = [Adam(lr = 0.0001), Adam(lr = 0.001), Adam(lr = 0.002), Adam(lr = 0.01)]
dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
batch_size = [128, 256, 512, 1024, 2048]
units = [4, 8, 16, 32, 64, 128, 256, 512]
dense1units = [4, 8, 16, 32, 64, 128, 256, 512]
dense2units = [4, 8, 16, 32, 64, 128, 256, 512]
dense3units = [4, 8, 16, 32, 64, 128, 256, 512]
param_grid = dict(batch_size = batch_size, optimizer = optimizer)
Kmodel = KerasClassifier(build_fn=getModel, verbose=1)
random = RandomizedSearchCV(estimator=Kmodel, param_distributions=param_grid, scoring='accuracy', n_jobs=-1, refit='boolean')
LSTM_result = random.fit(X_train,y_train)
LSTM_result.best_estimator_.get_params()