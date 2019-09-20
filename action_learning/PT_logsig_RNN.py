import time
import random
import iisignature
import numpy as np
from keras import layers
from keras.layers import Conv1D,Conv2D, Input, Lambda, Reshape, Permute, Flatten
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


def build_lin_Logsig_rnn_model(input_shape, n_hidden_neurons, output_shape, no_of_segments, deg_of_logsig, learning_rate, drop_rate_1,drop_rate_2, filter_size):
    """
    The LP_logsig_rnn model
    """
    logsiglen = iisignature.logsiglength(filter_size,deg_of_logsig)

    input_layer = Input(shape= input_shape)

    # Convolutional layer
    
    lin_projection_layer = Conv2D(32,(1,1), strides=(1,1), data_format='channels_last')(input_layer)
    lin_projection_layer = Conv2D(16,(5,1), strides=(1,1),data_format='channels_last')(lin_projection_layer)

    reshape = Reshape((input_shape[0]-4, 16*25))(lin_projection_layer)
    lin_projection_layer = Conv1D(filter_size,1)(reshape)
      
    mid_output = Lambda(lambda x: SP(x, no_of_segments), output_shape=(no_of_segments, filter_size), name='start_position')(lin_projection_layer)

    # Computing Logsig layer
    
    hidden_layer = Lambda(lambda x:CLF(x, no_of_segments, deg_of_logsig), \
                          output_shape=(no_of_segments,logsiglen),name='logsig')(lin_projection_layer)
    hidden_layer = Reshape((no_of_segments,logsiglen))(hidden_layer)
    # Batchnormalization
    BN_layer = BatchNormalization()(hidden_layer)
    
    mid_input = layers.concatenate([mid_output, BN_layer], axis=-1)
    
    # LSTM
    lstm_layer = LSTM(units=n_hidden_neurons, return_sequences=True)(mid_input)

    # Dropout
    drop_layer = Dropout(drop_rate_2)(lstm_layer)
    output_layer = Flatten()(drop_layer)
    output_layer = Dense(output_shape, activation='softmax')(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#     sgd = SGD(lr=learning_rate, decay=0.95,momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics = ['accuracy'])

    return model