import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import math_ops, array_ops, nn_ops, init_ops
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import Adam
from keras import backend as K
from keras import activations, initializers
from keras.models import load_model
from keras.engine.topology import Layer
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import random


import numpy as np
import time
global_i = 0

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True


class Model(object):
    """
    Model for learning Pendigit data

    learning_rate: learining rate

    training_iters: number of epochs

    batch size: batch size

    (n_steps, n_input): size of each sample

    deg_of_logsig: degree of log-signature

    X, Y: train data and label

    testX, testY: test data and label
    """
    def __init__(self, learning_rate, training_iters, batch_size,  n_input, n_steps, deg_of_logsig, X, Y, testX, testY):
        self.learning_rate = learning_rate
        self.training_iters = training_iters
        self.batch_size = batch_size
        self.n_input = n_input
        self.n_steps = n_steps
        self.deg_of_logsig = deg_of_logsig
        self.X = X 
        self.Y = Y
        self.testX = testX
        self.testY = testY


    def BuildModelKerasMn(self):
        start = time.time()
        model = Sequential()
        model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=(self.n_steps, self.n_input)))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(10, activation='softmax'))
        model.summary()
        # compile 
        model.compile(optimizer=Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
            loss='categorical_crossentropy', metrics=['accuracy'])
        # data
        no_of_samples = np.shape(self.X)[0]
        step = 1
        Loss = []
        Elapsed = []
        
        model.fit(self.X, self.Y, epochs=self.training_iters, batch_size=self.batch_size,shuffle=True,validation_data=(self.X, self.Y))
        
        # Calculate accuracy for test data
        score = model.evaluate(self.testX,self.testY)

        # Save the model
        file_suffix = str(self.n_steps)+'_segments_'+str(self.n_input)
        model_name = 'model_'+file_suffix+'.h5'
        model.save(model_name)
        print("Testing Loss:",  score[0])
        print("Testing Accuracy:", score[1])
        elapsed = time.time()-start
        """
        if self.deg_of_logsig==0:
            plt.plot(Elapsed, Loss, label='Time series')
        else:
            plt.plot(Elapsed, Loss, label='degree %d' %self.deg_of_logsig)
        """
        return {'Accuracy': score[1], 'Time': elapsed, 'NStep': step}
