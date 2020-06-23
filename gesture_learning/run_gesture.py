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

from data_generator import DataGenerator
from cus_layers import *
from LP_logsig_rnn import *

cha_train = np.load('cha_train.npy')
cha_val = np.load('cha_val.npy')
train_label = np.load('cha_train_label.npy')
val_label = np.load('cha_val_label.npy')

frame_nb = cha_train.shape[1]
joints_dim = int(cha_train.shape[2]/3)

n_hidden_neurons = 128
batch_size = 120 
learning_rate = 0.001
epochs = 500
number_of_segment = 4
deg_of_logsig = 2
drop_rate1 = 0.3
drop_rate2 = 0.5
filter_size = 30

output_shape = train_label.shape[1]
input_shape = [frame_nb, joints_dim,3]

model = build_lin_Logsig_rnn_model(input_shape, n_hidden_neurons, output_shape, 
                                   number_of_segment, deg_of_logsig, learning_rate, drop_rate1, drop_rate2, filter_size)

training_generator = DataGenerator(cha_train.reshape(-1,39,19,3), train_label, batch_size)



model_name = 'gesture_model_dr1%d_dr2%d_fs%d_dg%d.hdf5' %(drop_rate1,drop_rate2, filter_size, deg_of_logsig)

# Reduce learning rate if the loss does not reduce for 50 epochs
reduce_lr = ReduceLROnPlateau(monitor='loss', patience=50, verbose=1, factor=0.8, min_lr=0.000001)
# Save the best model only
mcp_save = ModelCheckpoint(model_name, save_best_only=True, monitor='acc', mode='auto')

start = time.time()

hist = model.fit_generator(training_generator, epochs=epochs, shuffle=True, verbose=1,#validation_data=(cha_val.reshape(-1,39,19,3), val_label),
                               use_multiprocessing = True, workers=6,
      callbacks = [ reduce_lr, mcp_save])

print(model.evaluate(cha_val.reshape(-1,frame_nb,joints_dim,3), val_label))
print((time.time()-start)/3600)                                   
                                   
                                   
                                   