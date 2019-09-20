import time
import random
import iisignature
import numpy as np
from keras import layers
from keras.layers import Conv1D, Input, Lambda, Reshape, Permute
from keras.models import Model
from tensorflow import keras  
from keras.optimizers import Adam
from keras.engine import InputSpec
from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization, Dropout
import tensorflow as tf
from functools import partial
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from cus_layers import *



def build_lin_Logsig_rnn_model(input_shape, n_hidden_neurons, output_shape, no_of_segments, deg_of_logsig, learning_rate, drop_rate1, drop_rate2, filter_size):
    """
	Construct the LP_logsig_rnn model using the customized operations CLF, Cat_T and PS from cus_layers.py
    """
    logsiglen = iisignature.logsiglength(filter_size,deg_of_logsig)

    input_layer = Input(shape= input_shape)
    # Time path concatenation
    cat_layer = Lambda(lambda x:Cat_T(x, input_shape[0]), output_shape=(input_shape[0], input_shape[1]+1))(input_layer)
    # Convolutional layer
    lin_projection_layer = Conv1D(filter_size,1)(cat_layer)
    # Dropout
    drop_layer_1 = Dropout(drop_rate1)(lin_projection_layer)
    # Cumulative sum
    ps_layer = Lambda(lambda x:PS(x), output_shape=(input_shape[0],filter_size))(drop_layer_1)
#     BN_layer_0 = BatchNormalization()(lin_projection_layer)
    # Computing Logsig layer
    hidden_layer_1 = Lambda(lambda x:CLF(x, no_of_segments, deg_of_logsig), \
                          output_shape=(no_of_segments,logsiglen))(ps_layer)
    hidden_layer_2 = Reshape((no_of_segments,logsiglen))(hidden_layer_1)
    # Batchnormalization
    BN_layer_1 = BatchNormalization()(hidden_layer_2)
    # LSTM
    lstm_layer = LSTM(units=n_hidden_neurons)(BN_layer_1)
#     BN_layer_2 = BatchNormalization()(lstm_layer)
    # Dropout
    drop_layer_2 = Dropout(drop_rate2)(lstm_layer)
    output_layer = Dense(output_shape, activation='softmax')(drop_layer_2)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics = ['accuracy'])

    return model
