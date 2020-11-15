Learning SDE Using Recurrent Neural Network with Log Signature Features


Structure
====================================

Directory:

data simulation:

`SDEdataSimulation.py`: simulate SDE with different driving paths; simulated data is saved in BM_paths.npy and output.npy

model:

`LSTM_Learning_Lib.py`: construct Recurrent Neural Network by Keras


`FeatureSetCalculation_Lib.py`: compute log signatures for simulated sample paths

Data simulation
====================================
```
python SDEdataSimulation -hurst <float>
```

Model Training
====================================
```
python train_logsig_rnn.py -number_of_segment <int> -deg_of_logsig <int> -hurst <float>
```

Settings:

	> parameters:
	'deg_of_sig': degree of log signature features; 
		when 'deg_of_sig'=0, it generates folded raw data and train RNN 
		 
		when 'deg_of_sig'>=2, it generates corresponding degree log signature features and train Logsig-RNN
	'number_of_segment': number of segment
	'learning_rate': learning_rate for Adam optimizer
	'training_iters': number of epochs to train the model
	'batch_size': batch size
	'display_size': the step to display training loss when training the model
	'n_hidden': number of neurals in hidden layer
	'error_tol': error tolerance as a threshold to stop training
	'test_len': the testing set size
