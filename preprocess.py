"""
File name: preprocess.py
author: Sheng Yang
Date Created: 2021/04/12

This file read in the original data, convert open, close, high, low, and trade volume
at each minute to wavelet coefficients, and store the coefficients
At each timestamp, the former window = 100 many minutes are considered, and the following
lag = 5 many minutes are examined for labeling trend.

Steps: 
1. standardize each column;
2. assign labels; 
3. for each 100 * 5 matrix, apply wavelet decomposition down to level 6 
(the max depth 100 could hold) for each column and flatten all coefficients. 
The output feature thus has 1 * 103 * 5 features (6 levels, 2 + 2 + 4 + 7 + 13 + 25 + 50 = 103).
"""

# load packages
import os 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pywt

# constants 
curr_folder = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(curr_folder, 'data/510050_1m.csv')                 # path to read from
save_path = os.path.join(curr_folder, 'preprocess/510050_1m_wavelets.csv')  # path to save to

# parameters 
window = 100                          # window to look back
lag = 5                               # number of minutes to look forward
th = 0.01                             # threshold for claiming stationarity

# hyperparameters 
mother_wavelet = 'db4'                # the wavelet we choose to decompose
use_cols = ['c', 'o', 'h', 'l', 'v']


def load_data():
    """load 510050_1m.csv into a dataframe"""
    print('Start Loading Data ...')
    tbl = pd.read_csv(file_path,
                      names=['d', 't', 'o', 'h', 'l', 'c', 'v'],
                      skiprows=1,
                      low_memory=False).interpolate() # only one place is missing a vol 
    print('Finish Loading')
    return tbl


def label_data(data, window, lag, th):
    """
    transform open, high, low, close, and vol to z-scores, and give labels according 
    to the window, lag, and threshold 
    """
    print('Giving Labels ...')
    standardized_data = StandardScaler().fit_transform(data[use_cols])
    X_raw, Y = [], []
    for t in range(standardized_data.shape[0] - window - lag):
        X_raw.append(standardized_data[t:t + window])
        # use movement of close prices to assign labels
        curr_close = standardized_data[t, 0]
        price_movement = \
            (standardized_data[t + window: t + window + lag, 0].mean() - curr_close) / curr_close
        # give labels 
        if price_movement > th:
            Y.append(1)
        elif price_movement < -th:
            Y.append(-1)
        else:
            Y.append(0)
    print('Finish Labeling')
    return np.array(X_raw), np.array(Y)


def dwt(x):
    """
    convert z-scores of the previous window many minutes at each timestamp into wavelet coefficients 
    """
    wavelet_coefs = []
    for i in range(x.shape[1]):
        wavelet_coefs.append(np.concatenate(pywt.wavedec(x[:, i], mother_wavelet)))
    return np.concatenate(wavelet_coefs)


def transform(X_raw):
    """
    convert the entire X into a wavelet coefficient dataset 
    """
    print('Obtaining New Features from Wavelet Coefficients ...')
    out = np.array([dwt(x) for x in X_raw])
    print('Finish Feature Engineering')
    return out


def save_preprocess(X, Y):
    """
    save all preprocessed data into a csv for training and testing purposes
    """
    new_feature = pd.DataFrame(X)
    new_feature['label'] = Y
    print('Saving New Features to a file ...')
    new_feature.to_csv(save_path, index=False)
    print('New Features have been saved to {}'.format(save_path))


if __name__ == '__main__':
    tbl = load_data()
    X_raw, Y = label_data(tbl, window, lag, th)
    X = transform(X_raw)
    save_preprocess(X, Y)
