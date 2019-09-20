Model for Pen-digit data


Introduction
====================================
Supported structures and features:
    -RNN with LSTM
        -Log signature features
        -Folded raw data
        -Raw data

Pen-digit data:
    https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/


Requirements
====================================
1. Python 3
2. Keras
3. Tensorflow
4. esig


Structure
====================================

Directory:

data preprocessing:

`GetSeqMnistData.py`: preprocessing pen-digit 

model:

`LSTM_Learning_Lib.py`: construct Recurrent Neural Network by Keras

`FeatureSetCalculation_Lib.py`: compute log signatures for simulated sample paths


Model Training
====================================
`Pen-digit_learning_example.ipynb`: notebook of Pen-digit learning

Settings:

	> parameters:
	'deg_of_sig': degree of log signature features; 
		when 'deg_of_sig'=0, it generates folded raw data; 
		when 'deg_of_sig'=1, it generates raw data; 
		when 'deg_of_sig'>=2, it generates corresponding degree log signature features
	'number_of_segment': number of segment
	'learning_rate': learning_rate for Adam optimizer
	'training_iters': number of epochs to train the model
	'batch_size': batch size


	
