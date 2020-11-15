import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras import backend as K
from keras import activations, initializers
from keras.models import load_model
from keras.engine.topology import Layer
import random


import numpy as np
import time


class Model(object):
    """
    Model for learning synthetic SDE data

    learning_rate: learining rate

    training_iters: number of epochs

    batch size: batch size

    (n_steps, n_input): size of each sample

    deg_of_logsig: degree of log-signature

    X, Y: data and label

    """

    def __init__(self, learning_rate, training_iters, batch_size, display_step, n_input, n_steps, n_hidden, n_classes, deg_of_logsig, X, Y, prefix):
        self.learning_rate = learning_rate
        self.training_iters = training_iters
        self.batch_size = batch_size
        self.display_step = display_step
        self.n_input = n_input
        self.n_steps = n_steps
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.deg_of_logsig = deg_of_logsig
        self.X = X
        self.Y = Y
        self.prefix = prefix

    # Keras model
    def BuildModelKeras(self, test_len, error_tol):
        start = time.time()
        model = Sequential()
        # build a LSTM RNN
        # model.add(ResidualKeras(units=self.n_hidden, input_shape=(self.n_steps, self.n_input)))
        model.add(LSTM(units=self.n_hidden, input_shape=(
            self.n_steps, self.n_input)))
        model.add(Dense(self.n_classes))
        # compile
        adam = Adam(self.learning_rate)
        model.compile(optimizer=adam, loss='mean_squared_error')
        # data
        no_of_samples = np.shape(self.X)[0]
        test_data = self.X[no_of_samples -
                           test_len:].reshape((-1, self.n_steps, self.n_input))
        test_label = self.Y[no_of_samples -
                            test_len:].reshape((-1, self.n_classes))
        step = 1
        cost = error_tol
        Loss = []
        Elapsed = []

        while ((step < self.training_iters) & (cost >= error_tol)):
            index = np.random.random_integers(
                0, no_of_samples - test_len, self.batch_size)
            batch_x = self.X[index, :]
            batch_y = self.Y[index].reshape((self.batch_size, 1))
            batch_x = batch_x.reshape(
                (self.batch_size, self.n_steps, self.n_input))
            cost = model.train_on_batch(batch_x, batch_y)
            pred = model.predict(batch_x, self.batch_size)
            if step % self.display_step == 0:
                Loss.append(cost)
                Elapsed.append((time.time() - start) / 3600)
                print("Iter " + str(step) +
                      ", Training Accuracy= " + "{:.7f}".format(cost))
            step += 1

        model.summary()
        # Calculate accuracy for test data
        result = model.test_on_batch(test_data, test_label)
        # Save the model
        model.save(self.prefix + 'model_logsig%d_segment%d.h5' %
                   (self.deg_of_logsig, self.n_steps))
        elapsed = time.time() - start

        test_pred = model.predict(test_data)

        return {'Loss': result, 'Time': elapsed, 'Pred': test_pred, 'model': model}
