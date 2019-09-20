import esig
import esig.tosig as ts
stream2logsig = ts.stream2logsig
stream2sig = ts.stream2sig
logsigdim = ts.logsigdim
sigdim = ts.sigdim
import numpy as np


def TimeJointPath(BM_path, T):
    """
    Concatenate the path with the time coordinate
    """
    number_of_timstep = BM_path.size
    dT = T/ (number_of_timstep -1)
    BM = np.zeros([number_of_timstep, 2], dtype = float)
    for i in range(1, number_of_timstep, 1):
        for k in range(0, 2, 1):
            if (k == 0): 
                BM[i, k] = dT+BM[i-1, k]
            if (k == 1): 
                BM[i, k] = BM_path[i]
    return BM

def ComputeMultiLevelSig(path, number_of_segment,  deg_of_sig):
    """
    compute the signature and log-signature of segments of one path of dimension (n, d)
    """
    n_t = path.shape[0]
    n_Path = path.shape[1]
    t_vec = np.arange(1, n_t, int(n_t/number_of_segment))
    t_vec = np.append(t_vec, n_t)
    MultiLevelSig = np.empty
    MultiLevelLogSig = np.empty
    
    for i in range(1, number_of_segment+1, 1): 
        temp_path = path[t_vec[i-1]-1:t_vec[i], 0:2]
        if deg_of_sig == 1:
            temp = stream2sig(temp_path, 2)
            
            temp = temp[0: (n_Path+1)]
            templog = temp[1:(n_Path+1)]
        else:
            temp = stream2sig(temp_path, deg_of_sig)
            templog = stream2logsig(temp_path, deg_of_sig)
            
        tempStart = (temp_path[0, 1])
        #print(temp)
        if (i == 1):
            MultiLevelSig = temp
            MultiLevelLogSig = templog
            MultiStart = np.array([tempStart])
        else:
            MultiLevelSig = np.concatenate((MultiLevelSig,temp), axis = 0)
            MultiLevelLogSig = np.concatenate((MultiLevelLogSig,templog), axis = 0)
            MultiStart = np.append(MultiStart, [tempStart])
    return {'MultiLevelSig': MultiLevelSig, 'MultiLevelLogSig': MultiLevelLogSig, 'MultiStart': MultiStart}


# Compute logsig
def ComputeMultiLevelLogsig1dBM(BM_paths, number_of_segment, depth_of_tensors, T):
    """
    Compute the log-signature of all samples
    """
    no_of_samples = np.shape(BM_paths)[0]
    
    if depth_of_tensors == 1:
        MultiLevelLogSigs = np.zeros([no_of_samples , 2 * number_of_segment], dtype=float)
    else:
        MultiLevelLogSigs = np.zeros([no_of_samples , logsigdim(2,depth_of_tensors)*number_of_segment], dtype=float)
    
    # MultiLevelLogSigs = np.zeros([no_of_samples , logsigdim(2,depth_of_tensors)*number_of_segment], dtype=float)
    MultiStart = np.zeros([no_of_samples , number_of_segment], dtype = float)
    for j in range(0, no_of_samples, 1):
        print(j)
        BM = TimeJointPath(BM_paths[j, :], T)
        result2 = ComputeMultiLevelSig(BM, number_of_segment, depth_of_tensors)
        #MultiLevelSigs[j] = result2['MultiLevelSig']
        #print(result2)
        MultiLevelLogSigs[j] = result2['MultiLevelLogSig']
        MultiStart[j] = result2['MultiStart']
    n_input = int(np.shape(MultiLevelLogSigs)[1]/number_of_segment)
    X_logsig_start = MultiLevelLogSigs.reshape((np.shape(MultiLevelLogSigs)[0], number_of_segment, n_input))
    batch_x0 = MultiStart.reshape((np.shape(MultiLevelLogSigs)[0], number_of_segment, 1))
    X_logsig_start = np.concatenate((X_logsig_start, batch_x0), axis = 2)
    return X_logsig_start

# Compute sig
def ComputeMultiLevelSig1dBM(BM_paths, number_of_segment, depth_of_tensors, T):
    """
    Compute the signature of all samples
    """

    no_of_samples = np.shape(BM_paths)[0]
    MultiLevelSigs = np.zeros([no_of_samples , sigdim(2,depth_of_tensors)*number_of_segment], dtype=float)
    for j in range(0, no_of_samples, 1):
        print(j)
        BM = TimeJointPath(BM_paths[j, :], T)
        result2 = ComputeMultiLevelSig(BM, number_of_segment, depth_of_tensors)
        MultiLevelSigs[j] = result2['MultiLevelSig']
    if number_of_segment > 1:
        n_input = int(np.shape(MultiLevelSigs)[1]/number_of_segment)
        X_sig = MultiLevelSigs.reshape((np.shape(MultiLevelSigs)[0], number_of_segment, n_input))
    else: 
        X_sig = MultiLevelSigs
    return X_sig