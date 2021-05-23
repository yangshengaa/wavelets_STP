# set up a trading scheme by wavelet decomposition, 
# compare the returns by different prediction models. 

# what should be the benchmark?

"""
set up a naive high-freqency trading strategy.

Rules: 
long when a signal 2 is given, short when 0 is given, and clear when 1 is given. 
Once an action is dealt, ignore signal for lag many mintues 
"""

# TODO: debugging: fixing indexing issues 

from preprocess import * 
from train_test import * 


# hypter parameter 
window = 240        # window to look back (for current dataset, an entire day)
lag = 5             # number of minutes to look forward
th = 0.001          # threshold for claiming stationarity

train_days = 180    # number of days as the training dataset 
test_days = 20      # number of days as testing dataset 


# TODO: modify comments 
def label_data_and_transform(data, window, lag, th, split_at):
    """
    split the partitioned dataset further into training the testing periods, 
    assign labels to the window, lag, and threshold 

    :param split_at: the index position of the start of the testing period 
    """
    # split train and test temporally 
    train_dataset, test_dataset = np.split(data, [split_at])
    # standardize separately 
    scaler = StandardScaler().fit(train_dataset)
    train_standardized = scaler.transform(train_dataset)
    test_standardized = scaler.transform(test_dataset)
    # merge together 
    standardized_data = np.append(train_standardized, test_standardized, axis=0)
    
    # give labels to training
    num_data = standardized_data.shape[0]
    X_train, X_test, Y_train = [], [], []
    for t in range(window, split_at):
        X_train.append(standardized_data[t - window:t])
        curr_close = standardized_data[t, 0]
        price_movement = (
            standardized_data[t: t + lag, 0].mean() - curr_close
        ) / curr_close
        if price_movement > th:
            Y_train.append(2)
        elif price_movement < -th:
            Y_train.append(0)
        else:
            Y_train.append(1)
    
    # obtain testing features 
    X_test = []
    for t in range(split_at, num_data):
        X_test.append(standardized_data[t - window:t])

    # combine and feed to wavelet transform
    X_raw = np.append(X_train, X_test)
    X = transform(np.array(X_raw))  # contains both training and testing periods 
    X_train, X_test = np.split(X, [split_at - window])

    # assign indices 
    train_idx = train_dataset.index 
    test_idx = test_dataset.index 
    X_train = pd.DataFrame(X_train, index=train_idx)
    Y_train = pd.Series(Y_train, index=train_idx)
    X_test = pd.DataFrame(X_test, index=test_idx)   # TODO: fix index alignment issue 
    return X_train, Y_train, X_test


def train_assign_direction_by_period(X_train, Y_train, X_test):
    """
    in each training/testing period, use XGBoost trained with X_train and Y_train to give 
    directions on X_test. 

    :param X_train: the flattened wavelet features at each timestamp 
    :param Y_train: the training labels (0, 1, and 2) for the training period 
    :param X_test: the flattened wavelet features at each timestamp for testing 
    """
    # pca 
    pca_transformer = PCA(n_components=20).fit(X_train)
    X_train = pca_transformer.transform(X_train)
    test_idx = X_test.index
    X_test = pca_transformer.transform(X_test)

    # train model 
    xgb_clf = XGBClassifier(
        objective='multi:softmax',
        use_label_encoder=False
        ).fit(X_train, Y_train, eval_metric='merror')

    # prediction
    Y_test_prediction = xgb_clf.predict(X_test)
    return pd.Series(Y_test_prediction, index=test_idx)


def train_assign_direction(raw_data, 
                           window=window, lag=lag, th=th, 
                           train_days=train_days, test_days=test_days
                          ):
    """
    Given the raw data, on a rolling basis, train on train_days many days 
    and trade for the next test_day many days. The trading signals are given 
    by XGBoost trained on the train_days many days of data. 

    :param raw_data: the data of the original csv 
    :return the decision after each minute  
    """ 
    data_to_use = raw_data[['c', 'h', 'l', 'v']]
    train_period, test_period = train_days * 241, test_days * 241
    num_chunk = (data_to_use.shape[0] - train_period) // test_period

    # obtain directions 
    direction = pd.Series([], dtype=int)
    for i in range(num_chunk + 1):
        print(f'Training period {i}')
        # obtain train and test dataset 
        train_start_idx = i * test_period
        train_end_idx = train_start_idx + train_period  # also the test start idx 
        test_end_idx = train_end_idx + test_period
        data_chunk = data_to_use.loc[train_start_idx:test_end_idx - 1]
        
        # standardize and transform 
        X_train, Y_train, X_test = label_data_and_transform(data_chunk, window, lag, th, train_period)
        
        # give directions 
        direction_curr_period = train_assign_direction_by_period(X_train, Y_train, X_test)
        print(direction_curr_period)
        direction = pd.concat([direction, direction_curr_period])
        print(f'Finish training period {i}')
    return direction


# TODO fix alignment issue 
def trade(raw_data, direction):
    """
    given the raw_data and computed direction predicted by XGBoost, 
    how does the strategy perform in reality? 

    :param raw_data: the raw_data read from the preprocess folder 
    :param direction: the direction (0, 1, or 2) computed by XGBoost 
    :return a series of net values guided by the XGBoost predictions
    """
    direction = pd.Series(direction)
    trade_start_idx = raw_data.shape[0] - direction.shape[0]  # should equal to window 
    print(trade_start_idx)
    trade_data = raw_data.loc[trade_start_idx:]
    
    min_ret = trade_data.c.pct_change()
    guided_min_ret = min_ret.reset_index(drop=True) * (direction - 1).shift(1)

    cum_ret = (min_ret + 1).cumprod()
    guided_cum_ret = (guided_min_ret + 1).cumprod()
    
    # make a plot 
    plt.plot(cum_ret)
    plt.plot(guided_cum_ret)
    plt.show()
    plt.savefig(os.path.join(report_path, 'naive_trading.png'))
    return cum_ret, guided_cum_ret


if __name__ == '__main__':
    raw_data = load_data()  # from preprocess 
    direction = train_assign_direction(raw_data)  
    trade(raw_data, direction)
