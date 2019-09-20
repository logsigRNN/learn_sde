import os
import re
import numpy as np
from math import pow
import matplotlib.pyplot as plt
import random
from FeatureSetCalculation_Lib import ComputeMultiLevelSig
import esig
import esig.tosig as ts
stream2logsig = ts.stream2logsig
stream2sig = ts.stream2sig
logsigdim = ts.logsigdim
sigdim = ts.sigdim
from itertools import groupby
import re
import random

"""
Pen-digit data loading and preprocessing. 
"""

def GetSeqPenDigit(Path = 'pendigits-orig.tra.txt'):
    """
    Load the original pen-digit data
    """
    path=Path
    penf=open(path)
    pen=penf.read()
    penf.close()
    penseq = pen.split('\n')
    penseq = [x for x in penseq if '.SEGMENT' not in x]
    label = []
    for i in range(len(penseq)):
        if '.COMMENT' in penseq[i]:
            tmplabel = re.split(r' ', penseq[i])
            tmplabel = int(tmplabel[1])
            label.append(tmplabel)
            penseq[i]='.'
    penseq = [list(g) for k,g in groupby(penseq,lambda x:x=='.') if not k]
    penseq = penseq[1:]
    for i in range(len(penseq)):
        penseq[i] = [x for x in penseq[i] if '.PEN' not in x]
        penseq[i] = [x for x in penseq[i] if '.DT' not in x]
        penseq[i] = [x for x in penseq[i] if x!='']
        for j in range(len(penseq[i])):
            penseq[i][j] = re.split(r' ', penseq[i][j])
            penseq[i][j] = [int(x) for x in penseq[i][j] if x!='']

    sampleClip = len(label)
    penlabel = np.zeros((sampleClip,10))
    count = 0;
    for x in label:
        if(count == sampleClip):
            break
        penlabel[count,x] = 1;
        count = count + 1;

    count = 0
    rawdata = [0]*sampleClip
    for seq in penseq:
        tmprd = []
        tmpx = [k[0] for k in seq]
        tmpy = [k[1] for k in seq]
        minx = min(tmpx)
        maxx = max(tmpx)
        miny = min(tmpy)
        maxy = max(tmpy)
        scale = float(2)/max([(maxx-minx),(maxy-miny)])
        list_x = [(x-(maxx+minx)/2)*scale for x in tmpx]
        list_y = [(y-(maxy+miny)/2)*scale for y in tmpy]

        for i in range(int(2*len(seq))):
            if i%2 == 0:
                tmprd.append(list_x[int(i/2)])
            else:
                tmprd.append(list_y[int((i-1)/2)])

        rawdata[count] = tmprd
        count = count + 1;

    return rawdata, penlabel

def GetSeqPenNorm(path):
    """
    Load the normalized Pen-digit data
    """
    penf=open(path)
    pen=penf.read()
    penf.close()
    penseq = pen.split('\n')
    penseq.pop()
    sampleClip = len(penseq)

    
    rawdata = [0]*sampleClip
    penlabel = np.zeros((sampleClip,10))
    print(sampleClip)
    for i in range(sampleClip):
        penseq[i]=re.split(r'[, ]',penseq[i])
        penseq[i] = [int(x) for x in penseq[i] if x!='']
        rawdata[i] = penseq[i][:-1]
        
        penlabel[i,penseq[i][-1]] = 1
    
    
    return rawdata, penlabel

def GetSeqPenNormCalLogSig(logSigDegree, number_of_segment, Path = 'pendigits.tra.txt'):
    """
    Compute the log-signature of normalized Pend-digit data
    """
    data, label = GetSeqPenNorm(Path)
    sampleClip = len(data)

    Multilogsig = np.zeros([sampleClip , logsigdim(2,logSigDegree)*number_of_segment], dtype=float)
    MultiStart = np.zeros([sampleClip , number_of_segment], dtype = float)
    """
    max_interval = 0
    for j in range(sampleClip):
        if max_interval < len(data[j]):
            max_interval = len(data[j])

    print(max_interval)
    tmpdata = np.zeros((sampleClip, max_interval))
    for sn in range(sampleClip):
        tmplen = len(data[sn])
        tmpdata[sn, :tmplen] = data[sn][:]
    """
    count = 0
    for seq in data:
        tmpx = seq[1::2]
        tmpy = seq[::2]

        logSigArray = np.zeros((len(tmpx),2))

        logSigArray[:,0] = tmpx
        logSigArray[:,1] = tmpy
        result2 = ComputeMultiLevelSig(logSigArray, number_of_segment, logSigDegree)
        Multilogsig[count] = result2['MultiLevelLogSig']
        MultiStart[count] = result2['MultiStart']
        count += 1

    n_input = int(np.shape(Multilogsig)[1]/number_of_segment)
    X_logsig_start = Multilogsig.reshape((np.shape(Multilogsig)[0], number_of_segment, n_input))
    batch_x0 = MultiStart.reshape((np.shape(Multilogsig)[0], number_of_segment, 1))
    X_logsig_start = np.concatenate((X_logsig_start, batch_x0), axis = 2)

    return X_logsig_start, label    



def GetSeqPenandCalLogSig(logSigDegree, number_of_segment, Path = 'pendigits-orig.tra.txt'):
    """
    Compute the log-signature of original Pen-digit data
    """
    data, label = GetSeqPenDigit(Path)
    sampleClip = len(data)

    Multilogsig = np.zeros([sampleClip , logsigdim(2,logSigDegree)*number_of_segment], dtype=float)
    MultiStart = np.zeros([sampleClip , number_of_segment], dtype = float)
    """
    max_interval = 0
    for j in range(sampleClip):
        if max_interval < len(data[j]):
            max_interval = len(data[j])

    print(max_interval)
    tmpdata = np.zeros((sampleClip, max_interval))
    for sn in range(sampleClip):
        tmplen = len(data[sn])
        tmpdata[sn, :tmplen] = data[sn][:]
    """
    count = 0
    for seq in data:
        tmpx = seq[1::2]
        tmpy = seq[::2]

        logSigArray = np.zeros((len(tmpx),2))

        logSigArray[:,0] = tmpx
        logSigArray[:,1] = tmpy
        result2 = ComputeMultiLevelSig(logSigArray, number_of_segment, logSigDegree)
        Multilogsig[count] = result2['MultiLevelLogSig']
        MultiStart[count] = result2['MultiStart']
        count += 1

    n_input = int(np.shape(Multilogsig)[1]/number_of_segment)
    X_logsig_start = Multilogsig.reshape((np.shape(Multilogsig)[0], number_of_segment, n_input))
    batch_x0 = MultiStart.reshape((np.shape(Multilogsig)[0], number_of_segment, 1))
    X_logsig_start = np.concatenate((X_logsig_start, batch_x0), axis = 2)

    return X_logsig_start, label

def GetSeqPenDel(logSigDegree, number_of_segment, droprate=0.1, Path = 'pendigits-orig.tra.txt'):
    """
    Compute the log-signature of original Pen-digit data with points randomly dropped 
    """
    data, label = GetSeqPenDigit(Path)
    sampleClip = len(data)

    Multilogsig = np.zeros([sampleClip , logsigdim(2,logSigDegree)*number_of_segment], dtype=float)
    MultiStart = np.zeros([sampleClip , number_of_segment], dtype = float)
    """
    max_interval = 0
    for j in range(sampleClip):
        if max_interval < len(data[j]):
            max_interval = len(data[j])

    print(max_interval)
    tmpdata = np.zeros((sampleClip, max_interval))
    for sn in range(sampleClip):
        tmplen = len(data[sn])
        tmpdata[sn, :tmplen] = data[sn][:]
    """
    count = 0
    for seq in data:
        tmpx = seq[1::2]
        tmpy = seq[::2]

        logSigArray = np.zeros((len(tmpx),2))


        logSigArray[:,0] = tmpx
        logSigArray[:,1] = tmpy
        index = random.sample(range(logSigArray.shape[0]), int(logSigArray.shape[0]*droprate))
        logSigArray = np.delete(logSigArray, index, 0)
        result2 = ComputeMultiLevelSig(logSigArray, number_of_segment, logSigDegree)
        Multilogsig[count] = result2['MultiLevelLogSig']
        MultiStart[count] = result2['MultiStart']
        count += 1

    n_input = int(np.shape(Multilogsig)[1]/number_of_segment)
    X_logsig_start = Multilogsig.reshape((np.shape(Multilogsig)[0], number_of_segment, n_input))
    batch_x0 = MultiStart.reshape((np.shape(Multilogsig)[0], number_of_segment, 1))
    X_logsig_start = np.concatenate((X_logsig_start, batch_x0), axis = 2)

    return X_logsig_start, label

def MnistLI2(list_x,list_y, n_sec=20):
    """
    Linear interpolation function for up sampling
    """
    FinerArray = np.array([]).reshape(0,2)
    for i in range(len(list_x)-1):
        if list_x[i]<list_x[i+1]:
            xvals = np.linspace(list_x[i], list_x[i+1], n_sec)
            xvals = np.delete(xvals, 0)
            xvals = np.delete(xvals, -1)
            yinterp = np.interp(xvals, list_x[i:i+2], list_y[i:i+2])
            xvals = np.insert(xvals, 0, list_x[i])
            yinterp = np.insert(yinterp, 0, list_y[i])
            temparray = np.concatenate((xvals,yinterp)).reshape(2,len(xvals)).transpose()
            temparray = sorted(temparray, key = myfun0)
        elif list_x[i]>list_x[i+1]:
            xvals = np.linspace(list_x[i+1], list_x[i], n_sec)
            xvals = np.delete(xvals, 0)
            xvals = np.delete(xvals, -1)
            yinterp = np.interp(xvals, [list_x[i+1],list_x[i]], [list_y[i+1],list_y[i]])
            xvals = np.append(xvals,list_x[i])
            yinterp = np.append(yinterp, list_y[i])
            temparray = np.concatenate((xvals,yinterp)).reshape(2,len(xvals)).transpose()
            temparray = sorted(temparray, key = myfun0, reverse = True)
        else:
            if list_y[i]<list_y[i+1]:
                yvals = np.linspace(list_y[i], list_y[i+1], n_sec)
                yvals = np.delete(yvals, 0)
                yvals = np.delete(yvals, -1)
                xinterp = np.interp(yvals, list_y[i:i+2], list_x[i:i+2])
                yvals = np.insert(yvals, 0, list_y[i])
                xinterp = np.insert(xinterp, 0, list_x[i])
                temparray = np.concatenate((xinterp,yvals)).reshape(2,len(xinterp)).transpose()
                temparray = sorted(temparray, key = myfun1)
            if list_y[i]>list_y[i+1]:
                yvals = np.linspace(list_y[i+1], list_y[i], n_sec)
                yvals = np.delete(yvals, 0)
                yvals = np.delete(yvals, -1)
                xinterp = np.interp(yvals, [list_y[i+1],list_y[i]], [list_x[i+1],list_x[i]])
                yvals = np.append(yvals, list_y[i])
                xinterp = np.append(xinterp, list_x[i])
                temparray = np.concatenate((xinterp,yvals)).reshape(2,len(xinterp)).transpose()
                temparray = sorted(temparray, key = myfun1, reverse = True)
            else:
                continue

        FinerArray = np.concatenate((FinerArray, temparray), axis=0)
    last = np.array([[list_x[-1], list_y[-1]]])
    FinerArray = np.concatenate((FinerArray, last))
    return FinerArray


def PenLI(data):
    """
    Up sampling implementation
    """

    sampleClip = len(data)   
    tmplist = []
    for seq in data:
        tmpx = seq[1::2]
        tmpy = seq[::2]

        # logSigArray = np.zeros((len(tmpx),2))

        # logSigArray[:,0] = tmpx
        # logSigArray[:,1] = tmpy
        logSigArray = MnistLI2(tmpx, tmpy)
        tmplist.append(logSigArray)

    return tmplist





def GetSeqPenLI(tmplist, label, logSigDegree, number_of_segment, Path = 'pendigits-orig.tra.txt'):
    """
    Compute the log-signature of up-sampled Pen-digit data
    """
    sampleClip = len(tmplist)     

    Multilogsig = np.zeros([sampleClip , logsigdim(2,logSigDegree)*number_of_segment], dtype=float)
    MultiStart = np.zeros([sampleClip , number_of_segment], dtype = float)
    
    count = 0
    for logSigArray in tmplist:
        # logSigArray = np.concatenate((logSigArray, np.zeros((3700-logSigArray.shape[0],2))))
        result2 = ComputeMultiLevelSig(logSigArray, number_of_segment, logSigDegree)
        
        Multilogsig[count] = result2['MultiLevelLogSig']
        MultiStart[count] = result2['MultiStart']
        count += 1   

    n_input = int(np.shape(Multilogsig)[1]/number_of_segment)
    X_logsig_start = Multilogsig.reshape((np.shape(Multilogsig)[0], number_of_segment, n_input))
    batch_x0 = MultiStart.reshape((np.shape(Multilogsig)[0], number_of_segment, 1))
    X_logsig_start = np.concatenate((X_logsig_start, batch_x0), axis = 2)

    return X_logsig_start, label


def GetSeqPenLIDel(tmplist, label, logSigDegree, number_of_segment, droprate=0.1, Path = 'pendigits-orig.tra.txt'):
    """
    Compute the log-signature of up-sampled Pen-digit data with points randomly dropped
    """

    sampleClip = len(tmplist)     

    Multilogsig = np.zeros([sampleClip , logsigdim(2,logSigDegree)*number_of_segment], dtype=float)
    MultiStart = np.zeros([sampleClip , number_of_segment], dtype = float)
    
    count = 0
    for logSigArray in tmplist:
        # result2 = ComputeMultiLevelSig(logSigArray, number_of_segment, logSigDegree)
        
        index = random.sample(range(logSigArray.shape[0]), int(logSigArray.shape[0]*droprate))
        logSigArray = np.delete(logSigArray, index, 0)
        
        result2 = ComputeMultiLevelSig(logSigArray, number_of_segment, logSigDegree)
        
        Multilogsig[count] = result2['MultiLevelLogSig']
        MultiStart[count] = result2['MultiStart']
        count += 1   

    n_input = int(np.shape(Multilogsig)[1]/number_of_segment)
    X_logsig_start = Multilogsig.reshape((np.shape(Multilogsig)[0], number_of_segment, n_input))
    batch_x0 = MultiStart.reshape((np.shape(Multilogsig)[0], number_of_segment, 1))
    X_logsig_start = np.concatenate((X_logsig_start, batch_x0), axis = 2)

    return X_logsig_start, label


def GetSeqPenRawDel(data, label):
    sampleClip = len(label)
    dataV = np.zeros((sampleClip, 462))
    for i in range(sampleClip):
        index = random.sample(range(len(data[i])), int(len(data[i])/5))
        tmpdata = np.array(data[i])
        tmpdata = np.delete(tmpdata, index)
        tmplen = len(tmpdata)
        dataV[i, :tmplen] =  tmpdata[:]
    dataV = dataV.reshape(sampleClip, int(462/2), 2)
    return dataV, label
