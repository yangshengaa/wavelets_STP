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
3. for each window * 4 matrix, apply wavelet decomposition down to maximum level of the desired mother wavelet; flatten them and obtain the feature vector 
4. store in preprocess folder

Output column name: 
for each column (close, high, low, and etc), we preserve the wavelet type (approx/detail) and level 
for instance: 
    c_A_5_15 means the approx coefficient at level 5 index 15 for close price
    v_D_3_33 means the detail coefficient at level 3 index 33 for trading volume

TODO: convert the original data to parquet; revise reading (only read some columns, maybe faster)
TODO: create a synthetic dataset that contains slight temporal dependencies, coupled with Gaussian WN

"""

# load packages
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pywt

# constants 
file_path = 'data/510050_1m.csv'               # path to read from
save_path = 'preprocess/preprocess.parquet'    # path to save to
concat = np.concatenate

# load parameters
with open('parameters/parameters.json', 'r') as f:
    param_dict = json.load(f)
    window, lag, th, train_days, test_days = param_dict.values()

# hyper-parameters
mother_wavelet = 'db4'           # the wavelet we choose to decompose
mother_wavelet = pywt.Wavelet(mother_wavelet)
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
    X, Y = [], []
    for t in range(standardized_data.shape[0] - window - lag):
        # obtain x 
        X_raw_t = standardized_data[t:t + window]
        X.append(dwt(X_raw_t))

        # use movement of close prices to assign labels
        curr_close = standardized_data[t + window, 0]
        price_movement = (
            standardized_data[t + window: t + window + lag, 0].mean() - curr_close
        ) / curr_close
        # give labels 
        if price_movement > th:
            Y.append(2)
        elif price_movement < -th:
            Y.append(0)
        else:
            Y.append(1)
    print('Finish Labeling')
    return np.array(X), np.array(Y)


def dwt(x):
    """
    convert z-scores of the previous window many minutes at each timestamp into wavelet coefficients

    :param x: each row in X_raw
    :return the flattened array
    """
    wavelet_coeff = []
    for i in range(x.shape[1]):
        wavelet_coeff.append(concat(pywt.wavedec(x[:, i], mother_wavelet)))
    return concat(wavelet_coeff)


def obtain_col_names(data_len=window, mother_wavelet=mother_wavelet):
    """
    fetch the column name for flatten wavelet features 

    :param data_len: the length of the data at each period, usually set to window 
    :param mother_wavelet: the wavelet we would like to use 
    """
    # coin a sequence to obtain the structure
    hypothetical_data = np.random.rand(data_len)
    max_level = pywt.dwt_max_level(data_len, mother_wavelet)
    hypothetical_wavelet_coeff = pywt.wavedec(hypothetical_data, mother_wavelet)
    length_each_level = [x.shape[0] for x in hypothetical_wavelet_coeff]
    # approx 
    approx_length = length_each_level[0]
    approx_names = [f'A_{max_level}_{i}' for i in range(approx_length)]
    # detail 
    detail_names = []
    for n in range(max_level, 0, -1):
        curr_idx = max_level + 1 - n
        curr_detail_length = length_each_level[curr_idx]
        curr_detail_names = [f'D_{n}_{i}' for i in range(curr_detail_length)]
        detail_names = detail_names + curr_detail_names
    return approx_names + detail_names
    

def make_wavelet_df_info(X, Y):
    """
    construct a dataframe with column names indicating the wavelet type (approx/detail) and level,
    and the label. to be saved later 

    :param X, Y: the processed X and y
    """
    # obtain column names 
    wavelet_names_each_col = obtain_col_names()
    wavelet_names = [] 
    for col in use_cols:
        wavelet_names = wavelet_names + ['{}_{}'.format(col, x) for x in wavelet_names_each_col]
    
    # new dataframe, convert to 32 for less storage (parquet does not support 16)
    new_feature = pd.DataFrame(
        X, 
        columns=wavelet_names
    ).assign(
        label=Y
    ).astype('float32').astype({'label': 'int8'})  
    return new_feature



def write_to_parquet(df):
    """
    write to parquet from the constructed df 
    """
    print('Saving New Features to files ...')
    df.to_parquet(save_path)
    print('New Features have been saved to {}'.format(save_path))


# run the following to obtain the preprocessed data 
if __name__ == '__main__':
    # load data 
    tbl = load_data()
    # give label and obtain wavelet features 
    X, Y = label_data(tbl, window, lag, th)
    # pack into a dataframe 
    df = make_wavelet_df_info(X, Y)
    # save to folder
    write_to_parquet(df)
