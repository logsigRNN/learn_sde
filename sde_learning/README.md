Learning SDE Using Recurrent Neural Network with Log Signature Features


Introduction
====================================
Supported structures and features:

    -RNN with LSTM
    
    -Log signature features
    
    -Folded raw data
    
    -Raw data
        
Data: synthetic SDE data


Requirements
====================================
1. Python 3
2. Keras
3. Tensorflow
4. esig


Structure
====================================

Directory:

data simulation:

`SDEdataSimulation.py`: simulate SDE with different driving paths; simulated data is saved in BM_paths.npy and output.npy

model:

`LSTM_Learning_Lib.py`: construct Recurrent Neural Network by Keras, Tensorflow


`FeatureSetCalculation_Lib.py`: compute log signatures for simulated sample paths


Model Training
====================================
`SDE_learning_example.ipynb`: notebook of SDE learning

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
	'display_size': the step to display training loss when training the model
	'n_hidden': number of neurals in hidden layer
	'error_tol': error tolerance as a threshold to stop training
	'test_len': the testing set size
	'pre_len': the predicting set size
