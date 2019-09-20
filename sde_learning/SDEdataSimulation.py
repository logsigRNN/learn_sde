## sigtools requires numpy; it consumes numpy arrays and emits numpy vectors
import numpy as np
from numpy import genfromtxt, array
from math import pow 
import os
import math
## sigtools has a subpackage sigtools.tosig that analyses time series data
import esig
import esig.tosig as ts
## fractional brownian motion package
from fbm import FBM,fbm
import matplotlib.pyplot as plt
stream2logsig = ts.stream2logsig
stream2sig = ts.stream2sig
logsigdim = ts.logsigdim
sigdim = ts.sigdim

from numpy.random import standard_normal
from numpy import array, zeros, sqrt, shape, convolve

# Milstein's method
def ComputeY(Y_last, dt, dB, step):
    ans = Y_last+(-np.pi*Y_last+np.sin(np.pi*step*dt))*dt+Y_last*dB+0.5*Y_last*(dB*dB-dt)
    return ans



number_of_independent_simulations = 2200
total_no_steps = 1001
maturity = 1


BM  = np.zeros([total_no_steps, 2], dtype=float)
# BM_paths - number_of_independent_simulations of one dimensional path
BM_paths = np.zeros([number_of_independent_simulations, total_no_steps], dtype=float)
output = np.zeros([number_of_independent_simulations], dtype = float)

# Simulate SDE with Brownian motion driving path
def SimulteSDEdatabm(number_of_timstep, T, sigma = 1):
    output = 0
    dT = T/(number_of_timstep - 1)
    BM = np.zeros([number_of_timstep, 2], dtype=float)
    print(dT)
    for i in range(1, number_of_timstep, 1):
        for k in range(0, 2, 1):
            if (k == 0): 
                BM[i, k] = dT+BM[i-1, k]
            if (k == 1): 
                dB = standard_normal()*sqrt(dT)
                BM[i, k] = sigma*dB+BM[i-1, k]
        output = ComputeY(output, dT, dB, i)
    return {'output':output, 'BM':BM} 

# Simulate SDE with fractional Brownian motion driving path
def SimulteSDEdatafbm(number_of_timstep, T, H=0.75):
    output = 0
    dT = T/(number_of_timstep-1)
    fBM = np.zeros([number_of_timstep, 2], dtype=float)
    temp_fbm = fbm(n=number_of_timstep-1, hurst=H, length=T, method='daviesharte')
    for i in range(1, number_of_timstep):
        for k in range(0,2):
            if (k == 0): 
                fBM[i, k] = dT+fBM[i-1, k]
            if (k == 1):
                dfB = temp_fbm[i]-temp_fbm[i-1]
                fBM[i, k] = temp_fbm[i]
        output = ComputeY(output, dT, dfB, i)
    return {'output':output, 'BM':fBM}


for j in range(0, number_of_independent_simulations, 1):
    print(j)
    result1 = SimulteSDEdatabm(total_no_steps, maturity)
    output[j] = result1['output']
    BM = result1['BM']
    BM_paths[j] = np.transpose(BM[:, 1])
   

np.save('output', output)
np.save('BM_paths', BM_paths)

