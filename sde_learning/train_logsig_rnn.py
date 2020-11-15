import numpy as np
from LSTM_Learning_Lib import Model
from FeatureSetCalculation_Lib import ComputeMultiLevelLogsig1dBM
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import ParameterGrid
from sklearn import preprocessing
import random
import os


def main(prefix,
         BM_paths,
         output,
         deg_of_logsig,
         number_of_segment,
         lr=0.001,
         training_iters=8000000,
         batch_size=64,
         display_step=100,
         n_hidden=64,
         n_classes=1,
         error_tol=0.001 / 512,
         test_len=200,
         maturity=10):

    if not os.path.exists(prefix):
        # if the experiment directory does not exist we create the directory
        os.makedirs(prefix)
    T = maturity
    assert deg_of_logsig != 1

    Y = output
    sig_comp_time = []
    test_result = []
    test_time = []

    start = time.time()
    # RNN
    if deg_of_logsig == 0:
        n_input = 1
        X_raw = BM_paths.reshape(-1, BM_paths.shape[1], n_input)
        print(X_raw.shape)
        model3 = Model(lr, training_iters, batch_size, display_step,
                       n_input, X_raw.shape[1], n_hidden, n_classes, deg_of_logsig, X_raw, Y, prefix)
    # LogsigRNN
    else:
        X_logsig_start = ComputeMultiLevelLogsig1dBM(
            BM_paths, number_of_segment, deg_of_logsig, T)
        n_input = np.shape(X_logsig_start)[2]
        elapsed = time.time() - start
        sig_comp_time.append(elapsed)
        np.save(prefix + 'logsig_deg%d_ns%d.npy' %
                (number_of_segment, deg_of_logsig), X_logsig_start)
        model3 = Model(lr, training_iters, batch_size, display_step, n_input,
                       number_of_segment, n_hidden, n_classes, deg_of_logsig, X_logsig_start, Y, prefix)
    # build and train model
    print('start training:')
    fixed_error_result_model3 = model3.BuildModelKeras(test_len, error_tol)

    print("Time = " + str(time.time() - start))
    print("Testing loss = " + str(fixed_error_result_model3['Loss']))
    # model3.KerasPredict()
    test_result.append(fixed_error_result_model3['Loss'])
    test_time.append(fixed_error_result_model3['Time'])

    np.save(prefix + 'ns' + str(number_of_segment) + 'deg_logsig' + str(
        deg_of_logsig) + '_test_result', test_result)
    np.save(prefix + 'ns' + str(number_of_segment) +
            'deg_logsig' + str(deg_of_logsig) + '_test_time', test_time)
    np.save(prefix + 'ns' + str(number_of_segment) + 'deg_logsig' + str(
        deg_of_logsig) + '_sig_comp_time', elapsed)

    return fixed_error_result_model3['Loss'], fixed_error_result_model3['model'], fixed_error_result_model3['Pred']


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-number_of_segment', default=8, type=int)
    parser.add_argument('-deg_of_logsig', default=4, type=int)
    parser.add_argument('-hurst', default=0.25, type=float)
    args = parser.parse_args()

    prefix = 'results/fbm_%.2f/' % args.hurst
    BM_paths = np.load('data/fbm_%.2f/paths.npy' % args.hurst)
    number_of_samples = BM_paths.shape[0]
    """
    BM1001 = np.zeros([number_of_samples, 1001])
    for i in range(number_of_samples):
        for j in range(50001):
            if j%50==0:
                BM1001[i][int(j/50)] = BM_paths[i][j]
    BM_paths = BM1001
    """

    output = np.load('data/fbm_%.2f/output.npy' % args.hurst)

    test_loss, trained_model, test_pred = main(prefix,
                                               BM_paths,
                                               output,
                                               args.deg_of_logsig,
                                               args.number_of_segment,)
