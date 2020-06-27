# Learning SDE Using Recurrent Neural Network with Log Signature Features
Using logsignatures as features or layers in a recurrent neural network

This code is for paper Learning stochastic differential equations using RNN with log signature features

## Directory
1. `sde_learning`: Codes for synthetic SDE data learning

2. `Pen-digit_learning`: Codes for [Pen-digit data](https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/) learning

3. `gesture_learning`: Codes for [Chalearn 2013 gesture data](http://sunai.uoc.edu/chalearn/) learning

4. `action_learning`: Codes for [NTURGB+D 120](https://github.com/shahroudy/NTURGB-D) learning

## Dependencies
Python 3.7 was used. The following external packages were used, and may be installed via `pip3 install -r requirements.txt`.

[`iisignature==0.23`](https://github.com/bottler/iisignature) for calculating signatures. It is CPU-only; consequently signature-based models are currently bottlenecked here.

[`jupyter==1.0.0`](https://jupyter.org/)

[`Keras==2.2.4`](https://github.com/keras-team/keras.git)

[`esig==0.6.31`](https://pypi.org/project/esig/0.6.31/)

[`matplotlib==2.2.4`](https://matplotlib.org/)

[`scikit-learn==0.20.3`](https://scikit-learn.org/)
