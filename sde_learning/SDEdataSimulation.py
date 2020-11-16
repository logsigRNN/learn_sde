# sigtools requires numpy; it consumes numpy arrays and emits numpy vectors
import numpy as np
from numpy.random import standard_normal
import os
import math
from tqdm import tqdm, trange
# fractional brownian motion package
from fbm import FBM, fbm


# Milstein's method
def ComputeY(Y_last, dt, dB, step):
    ans = Y_last + (-np.pi * Y_last + np.sin(np.pi * step * dt)) * \
        dt + Y_last * dB + 0.5 * Y_last * (dB * dB - dt)
    return ans

# Simulate SDE with Brownian motion driving path


def SimulteSDEdatabm(number_of_timstep, T, sigma=1):
    dT = T / (number_of_timstep - 1)
    BM = np.zeros([number_of_timstep, 2], dtype=float)
    output = [0]
    for i in range(1, number_of_timstep, 1):
        for k in range(0, 2, 1):
            if (k == 0):
                BM[i, k] = dT + BM[i - 1, k]
            if (k == 1):
                dB = standard_normal() * math.sqrt(dT)
                BM[i, k] = sigma * dB + BM[i - 1, k]
        output.append(ComputeY(output[-1], dT, dB, i))
    return {'output': np.array(output), 'BM': BM}

# Simulate SDE with fractional Brownian motion driving path


def SimulteSDEdatafbm(number_of_timstep, T, H=0.75):
    dT = T / (number_of_timstep - 1)
    fBM = np.zeros([number_of_timstep, 2], dtype=float)
    output = [0]
    temp_fbm = fbm(n=number_of_timstep - 1, hurst=H,
                   length=T, method='daviesharte')
    for i in range(1, number_of_timstep):
        for k in range(0, 2):
            if (k == 0):
                fBM[i, k] = dT + fBM[i - 1, k]
            if (k == 1):
                dfB = temp_fbm[i] - temp_fbm[i - 1]
                fBM[i, k] = temp_fbm[i]
        output.append(ComputeY(output[-1], dT, dfB, i))
    return {'output': np.array(output), 'BM': fBM}


def get_sde_paths(hurst, prefix, number_of_independent_simulations=2200, total_no_steps=50001, maturity=10):
    if not os.path.exists(prefix):
        # if the experiment directory does not exist we create the directory
        os.makedirs(prefix)

    BM = np.zeros([total_no_steps, 2], dtype=float)
    # BM_paths - number_of_independent_simulations of one dimensional path
    BM_paths = np.zeros([number_of_independent_simulations,
                         total_no_steps], dtype=float)
    output = np.zeros([number_of_independent_simulations,
                       total_no_steps], dtype=float)

    for j in tqdm(range(0, number_of_independent_simulations, 1), total=number_of_independent_simulations):
        if hurst == 0.5:
            result1 = SimulteSDEdatabm(total_no_steps, maturity)
        else:
            result1 = SimulteSDEdatafbm(total_no_steps, maturity, hurst)
        output[j] = result1['output']
        BM = result1['BM']
        BM_paths[j] = np.transpose(BM[:, 1])

    np.save(prefix + 'output', output)
    np.save(prefix + 'paths', BM_paths)
    return BM_paths, output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-hurst', default=0.25, type=float)
    args = parser.parse_args()
    prefix = 'data/fbm_%.2f/' % args.hurst

    BM_paths, output = get_sde_paths(args.husrt, prefix)
