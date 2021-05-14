# TODO: update file header 

"""
File name: preprocess.py
author: Sheng Yang
Date Created: 2021/04/12

This file read in the original data, convert open, close, high, low, and trade volume
at each minute to wavelet coefficients, and store the coefficients
At each timestamp, the former window = 240 many minutes are considered, and the following
lag = 5 many minutes are examined for labeling trend.

Steps: 
1. standardize each column;
2. assign labels; 
3. for each window * 5 matrix, apply wavelet decomposition down to maximum level of the desired mother wavelet
"""

# load packages
import os
import numpy as np
import pandas as pd
from multiprocessing import Process
import numba
from sklearn.preprocessing import StandardScaler
import pywt

# constants 
curr_folder = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(curr_folder, 'data/510050_1m.csv')  # path to read from
save_path = 'preprocess'  # path to save to

# parameters 
window = 240  # window to look back (for current dataset, an entire day)
lag = 5  # number of minutes to look forward
th = 0.01  # threshold for claiming stationarity

# hyper-parameters
mother_wavelet = 'db4'  # the wavelet we choose to decompose
use_cols = ['c', 'h', 'l', 'v']  # discard open


def load_data():
    """ load the raw csv file """
    print('Start Loading Data ...')
    df = pd.read_csv(file_path,
                     names=['d', 't', 'o', 'h', 'l', 'c', 'v'],
                     skiprows=1,
                     low_memory=False).interpolate()  # only one place is missing a vol
    print('Finish Loading')
    return df


def label_data(data, window, lag, th):
    """
    transform open, high, low, close, and vol to z-scores, and give labels according 
    to the window, lag, and threshold

    :param data: the read in dataframe
    :param window: the window to look back (240 e.g.)
    :param lag: the look forward period (5 e.g.)
    :param th: the threshold for assigning direction. 2 for >th, 0 for <-th, and 1 otherwise
    :return the raw X and label y
    """
    print('Giving Labels ...')
    standardized_data = StandardScaler().fit_transform(data[use_cols])
    X_raw, Y = [], []
    for t in range(standardized_data.shape[0] - window - lag):
        X_raw.append(standardized_data[t:t + window])
        # use movement of close prices to assign labels
        curr_close = standardized_data[t, 0]
        price_movement = (standardized_data[t + window: t + window + lag, 0].mean() - curr_close) / curr_close
        # give labels 
        if price_movement > th:
            Y.append(2)
        elif price_movement < -th:
            Y.append(0)
        else:
            Y.append(1)
    print('Finish Labeling')
    return np.array(X_raw), np.array(Y)


def dwt(x):
    """
    convert z-scores of the previous window many minutes at each timestamp into wavelet coefficients

    :param x: each row in X_raw
    :return the flattened array
    """
    wavelet_coefs = []
    for i in range(x.shape[1]):
        wavelet_coefs.append(np.concatenate(pywt.wavedec(x[:, i], mother_wavelet)))
    return np.concatenate(wavelet_coefs)


def transform(X_raw):
    """
    convert the entire X into a wavelet coefficient dataset

    :param X_raw: the standardized X from the original dataframe
    :return the wavelet features flattened in each row
    """
    # print('Obtaining New Features from Wavelet Coefficients ...')
    out = np.array([dwt(x) for x in X_raw])
    # print('Finish Feature Engineering')
    return out


def save_preprocess(X, Y, save_path='preprocess', n_rows=40000):
    """
    save all preprocessed data into a csv for training and testing purposes

    :param X, Y: the processed X and y
    :param save_path: the path/folder to be saved to
    :param n_rows: the number of rows to write at each time
    """
    # convert to 16 for less storage
    new_feature = pd.DataFrame(X).assign(label=Y).astype('float16').astype({'label': 'int8'})
    print('Saving New Features to files ...')

    n_files = new_feature.shape[0] // n_rows
    # split into a couple of files
    p_list = []
    for i in range(n_files + 1):
        p_list.append(
            Process(target=write_to_file_helper,
                    args=(
                        new_feature.iloc[(i * n_rows):((i + 1) * n_rows)],
                        os.path.join(save_path, f'510050_1m_wavelets_{i}.csv')
                    )
                    )
        )
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()
    print('New Features have been saved to {}'.format(save_path))


def write_to_file_helper(df, name):
    """ write file helper for multiprocessing """
    df.to_csv(name, index=False, header=False)


if __name__ == '__main__':
    tbl = load_data()
    X_raw, Y = label_data(tbl, window, lag, th)
    X = transform(X_raw)
    save_preprocess(X, Y)
