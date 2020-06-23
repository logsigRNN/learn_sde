import time
import random
import iisignature
import numpy as np
from keras import layers
from keras.layers import Conv2D,Conv1D, Input, Lambda, Reshape, Permute,Flatten
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
    logsiglen = iisignature.logsiglength(filter_size+1,deg_of_logsig)
    input_layer = Input(shape= input_shape)
    lin_projection_layer = Conv2D(32,(1,1), strides=(1,1), data_format='channels_last')(input_layer)
    lin_projection_layer = Conv2D(16,(3,1), strides=(1,1), padding='same',data_format='channels_last')(lin_projection_layer)
    reshape = Reshape((input_shape[0], 16*19))(lin_projection_layer)

    lin_projection_layer = Conv1D(filter_size,1,activation='relu')(reshape)
    drop_layer_1 = Dropout(drop_rate_1)(lin_projection_layer)
    ps_layer = Lambda(lambda x:PS(x), output_shape=(input_shape[0],filter_size),name='partial_sum')(drop_layer_1)
    cat_layer = Lambda(lambda x:Cat_T(x,input_shape[0]), output_shape=(input_shape[0], filter_size+1),name='add_time')(ps_layer)
    
    mid_output = Lambda(lambda x: SP(x, no_of_segments), output_shape=(no_of_segments, filter_size+1), name='start_position')(cat_layer)

    hidden_layer_1 = Lambda(lambda x:CLF(x, no_of_segments, deg_of_logsig), \
                          output_shape=(no_of_segments,logsiglen), name='logsig_layer')(cat_layer)
    hidden_layer_2 = Reshape((no_of_segments,logsiglen))(hidden_layer_1)
    BN_layer_1 = BatchNormalization()(hidden_layer_2)

    mid_input = layers.concatenate([mid_output, BN_layer_1], axis=-1)

    lstm_layer = LSTM(units=n_hidden_neurons, return_sequences=True)(mid_input)
#     hidden_layer_3 = Dense(512, activation='relu')(lstm_layer)
#     BN_layer_2 = BatchNormalization()(lstm_layer)
    drop_layer_2 = Dropout(drop_rate_2)(lstm_layer)
    output_layer = Flatten()(drop_layer_2)
    output_layer = Dense(output_shape, activation='softmax')(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics = ['accuracy'])

    return model

