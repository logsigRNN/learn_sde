import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import math_ops, array_ops, nn_ops, init_ops
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras import backend as K
from keras import activations, initializers
from keras.models import load_model
from keras.engine.topology import Layer
from sklearn.model_selection import StratifiedKFold
import random


import numpy as np
import time
global_i = 0

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
#session = tf.Session(config=config, ...)



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
    def __init__(self, learning_rate, training_iters, batch_size, display_step, n_input, n_steps, n_hidden, n_classes,deg_of_sig, X, Y):
        self.learning_rate = learning_rate
        self.training_iters = training_iters
        self.batch_size = batch_size
        self.display_step = display_step
        self.n_input = n_input
        self.n_steps = n_steps
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.deg_of_sig = deg_of_sig
        self.X = X 
        self.Y = Y

        # Define weights
        self.weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

    # Tensorflow
    def BuildModelTF(self,  test_len, error_tol):
            # tf Graph input
            start = time.time()
            x = tf.placeholder("float", [None, self.n_steps, self.n_input])
            y = tf.placeholder("float", [None, self.n_classes])
            x1 = tf.unstack(x, self.n_steps, 1)
            lstm_cell = rnn.LSTMCell(self.n_hidden, forget_bias=1.0)
            # rrn_cell = ResidualCell(self.n_hidden, reuse=tf.AUTO_REUSE)
            global global_i
            if (global_i>0):
                with tf.variable_scope( "rnn/basic_lstm_cell"+str(global_i), reuse = False):
                # with tf.variable_scope("rnn/residual_cell"+str(global_i), reuse = False):
                    weights = tf.get_variable("weights", [self.n_hidden, self.n_classes])
                    outputs, states = rnn.static_rnn(lstm_cell, x1, dtype=tf.float32)
                    # outputs, states = rnn.static_rnn(rrn_cell, x1, dtype=tf.float32)
            else:
                outputs, states = rnn.static_rnn(lstm_cell, x1, dtype=tf.float32)
                # outputs, states = rnn.static_rnn(rrn_cell, x1, dtype=tf.float32)
            pred = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']
            
            # Define loss and optimizer
            cost = tf.losses.mean_squared_error(y,pred)
            # cost = tf.nn.l2_loss(pred - y)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
            # Initializing the variables
            init = tf.global_variables_initializer()
                    # Launch the graph
            no_of_samples = np.shape(self.X)[0]
            #temp = self.X[:test_len]
            test_data = self.X[no_of_samples-test_len:].reshape((-1, self.n_steps, self.n_input)) 
            test_label = self.Y[no_of_samples-test_len:].reshape((-1,self.n_classes))
            #result = []
            Loss=[]
            Elapsed = []
            Iter = []
            loss = error_tol
            with tf.Session(config=config) as sess:
                sess.run(init)
                step = 1
                # Keep training until reach error tolerance or max iterations
                while ((step  < self.training_iters) & (loss >= error_tol)):
                    index = np.random.random_integers(0, no_of_samples-test_len, self.batch_size)
                    batch_x = self.X[index, :]
                    batch_y = self.Y[index].reshape((self.batch_size, 1))
                    batch_x = batch_x.reshape((self.batch_size, self.n_steps, self.n_input))
                    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})  
                    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})           
                    if step % self.display_step == 0:
                        print("Iter " + str(step) + ", Minibatch Loss= " +                   "{:.6f}".format(loss) )
                        Loss.append(loss)
                        Elapsed.append((time.time()-start)/3600)
                        Iter.append(step)
                    step += 1

            # Calculate loss for test data         
                result = sess.run(cost, feed_dict={x: test_data, y: test_label})
                test_pred = sess.run(pred, feed_dict={x: test_data, y: test_label})
                pred_save = []
                for k in range(len(test_pred)):
                    pred_save.append(test_pred[k][0])
                f = open('pred.txt','w+')
                print(pred_save, file = f)
                elapsed = time.time()-start
            """
            if self.deg_of_sig==0:
                plt.plot(Iter, Loss, label='Time series')
            else:
                plt.plot(Iter, Loss, label='degree %d' %self.deg_of_sig)
            """
            global_i = global_i+1
            return {'Loss': result, 'Time': elapsed, 'NStep': step}

    # K-fold Cross Validation for Tensorflow
    def BuildModelTFCV(self,  test_len, error_tol):
            # tf Graph input
            x = tf.placeholder("float", [None, self.n_steps, self.n_input])
            y = tf.placeholder("float", [None, self.n_classes])
            x1 = tf.unstack(x, self.n_steps, 1)
            lstm_cell = rnn.LSTMCell(self.n_hidden, forget_bias=1.0)
            # rrn_cell = ResidualCell(self.n_hidden, reuse=tf.AUTO_REUSE)
            global global_i
            if (global_i>0):
                with tf.variable_scope( "rnn/basic_lstm_cell"+str(global_i), reuse = False):
                # with tf.variable_scope("rnn/residual_cell"+str(global_i), reuse = False):
                    weights = tf.get_variable("weights", [self.n_hidden, self.n_classes])
                    outputs, states = rnn.static_rnn(lstm_cell, x1, dtype=tf.float32)
                    # outputs, states = rnn.static_rnn(rrn_cell, x1, dtype=tf.float32)
            else:
                outputs, states = rnn.static_rnn(lstm_cell, x1, dtype=tf.float32)
                # outputs, states = rnn.static_rnn(rrn_cell, x1, dtype=tf.float32)
            pred = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']
            
            # Define loss and optimizer
            cost = tf.losses.mean_squared_error(y,pred)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
            # Initializing the variables
            init = tf.global_variables_initializer()
                    # Launch the graph
            #temp = self.X[:test_len]
            #result = []
            Loss=[]
            Elapsed = []
            Iter = []
            with tf.Session(config=config) as sess:
                kfold = StratifiedKFold(n_splits=10)
                cvscores = []
                cvtime = []
                for traindex, testindex in kfold.split(self.X, self.Y.astype('int')):
                    start = time.time()
                    trainX = self.X[traindex]
                    trainY = self.Y[traindex]
                    testX = self.X[testindex].reshape((-1, self.n_steps, self.n_input))
                    testY = self.Y[testindex].reshape((-1,self.n_classes))
                    sess.run(init)
                    step = 1
                    loss = error_tol
                    # Keep training until reach error tolerance or max iterations
                    while ((step  < self.training_iters) & (loss >= error_tol)):
                        index = np.random.random_integers(0,trainX.shape[0]-1,self.batch_size)
                        batch_x = trainX[index, :]
                        batch_y = trainY[index].reshape((self.batch_size, 1))                                           
                        batch_x = batch_x.reshape((self.batch_size, self.n_steps, self.n_input))
                        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})  
                        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})           
                        if step % self.display_step == 0:
                            print("Iter " + str(step) + ", Minibatch Loss= " +                   "{:.6f}".format(loss) )
                            Loss.append(loss)
                            Elapsed.append((time.time()-start)/3600)
                            Iter.append(step)
                        step += 1
                # Calculate accuracy for test data           
                    result = sess.run(cost, feed_dict={x: testX, y: testY})
                    test_pred = sess.run(pred, feed_dict={x: testX, y: testY})
                    print("Testing Loss:",  result)
                    cvscores.append(result)
                    elapsed = time.time()-start
                    cvtime.append(elapsed)
                    
            """
            if self.deg_of_sig==0:
                plt.plot(Iter, Loss, label='Time series')
            else:
                plt.plot(Iter, Loss, label='degree %d' %self.deg_of_sig)
            """

            global_i = global_i+1
            return {'Loss': cvscores, 'Time': np.mean(cvtime), 'NStep': step}

    
    # Keras model
    def BuildModelKeras(self,  test_len, error_tol):
        start = time.time()
        model = Sequential()
        # build a LSTM RNN
        # model.add(ResidualKeras(units=self.n_hidden, input_shape=(self.n_steps, self.n_input)))
        model.add(LSTM(units=self.n_hidden, input_shape=(self.n_steps, self.n_input)))
        model.add(Dense(self.n_classes))
        # compile 
        adam = Adam(self.learning_rate)
        model.compile(optimizer=adam,loss='mean_squared_error')
        # data
        no_of_samples = np.shape(self.X)[0]
        test_data = self.X[no_of_samples-test_len:].reshape((-1, self.n_steps, self.n_input)) 
        test_label = self.Y[no_of_samples-test_len:].reshape((-1,self.n_classes))
        step = 1
        cost = error_tol
        Loss = []
        Elapsed = []
        
        while ((step  < self.training_iters) & (cost >= error_tol)):
            index = np.random.random_integers(0, no_of_samples-test_len, self.batch_size)
            batch_x = self.X[index, :]
            batch_y = self.Y[index].reshape((self.batch_size, 1))
            batch_x = batch_x.reshape((self.batch_size, self.n_steps, self.n_input))
            cost = model.train_on_batch(batch_x, batch_y)
            pred = model.predict(batch_x, self.batch_size)
            print("Iter " + str(step) + ", Training Accuracy= " +                   "{:.7f}".format(cost))
            if step % self.display_step == 0:
                Loss.append(cost)
                Elapsed.append((time.time()-start)/3600)
                print("Iter " + str(step) + ", Training Accuracy= " +                   "{:.7f}".format(cost))
            step+=1
        """
        train_x = self.X.reshape((2000, self.n_steps, self.n_input))
        train_y = self.Y.reshape((2000,1))
        model.fit(train_x, train_y, epochs=5000, batch_size=self.batch_size)
        """
        model.summary()
        # Calculate accuracy for test data
        result = model.test_on_batch(test_data,test_label)
        # Save the model
        model.save('Kerasmodel.h5')
        print("Testing Loss:",  result)
        elapsed = time.time()-start
        """
        if self.deg_of_sig==0:
            plt.plot(Elapsed, Loss, label='Time series')
        else:
            plt.plot(Elapsed, Loss, label='degree %d' %self.deg_of_sig)
        """
        return {'Loss': result, 'Time': elapsed, 'NStep': step}

