# set up a trading scheme by wavelet decomposition, 
# compare the returns by different prediction models. 

# what should be the benchmark?

"""
set up a naive high-freqency trading strategy.

Rules: 
long when a signal 2 is given, short when 0 is given, and clear when 1 is given. 
Once an action is dealt, ignore signal for lag many mintues 
"""

# TODO: debugging

from preprocess import * 
from train_test import * 


# hypter parameter 
window = 240  # window to look back (for current dataset, an entire day)
lag = 5  # number of minutes to look forward
th = 0.01  # threshold for claiming stationarity

train_days = 360    # number of days as the training dataset 
test_days = 60      # number of days as testing dataset 


# TODO: modify comments 
def label_data_and_transform(data, window, lag, th, split_at):
    """
    standardize high, low, close, and vol based on training period 
    and transform both training and testing periods. Then, give labels according 
    to the window, lag, and threshold

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
    
    # give labels and split 
    X_raw, Y = [], []
    for t in range(standardized_data.shape[0] - window - lag):
        X_raw.append(standardized_data[t:t + window])
        # use movement of close prices to assign labels
        curr_close = standardized_data[t, 0]
        price_movement = (
            standardized_data[t + window: t + window + lag, 0].mean() - curr_close) / curr_close
        # give labels
        if price_movement > th:
            Y.append(2)
        elif price_movement < -th:
            Y.append(0)
        else:
            Y.append(1)
    
    # obtain train and test dataset 
    X = transform(np.array(X_raw))  # contains both training and testing periods 
    X_train, X_test = np.split(X, [split_at])
    Y_train, _ = np.split(Y, [split_at])      # contains training periods only 

    # assign indices 
    train_idx = train_dataset.index 
    test_idx = test_dataset.index 
    X_train = pd.DataFrame(X_train, index=train_idx)
    Y_train = pd.Series(Y_train, index=train_idx)
    X_test = pd.DataFrame(X_test, index=test_idx)   # TODO: fix index alignment issue 
    return X_train, Y_train, X_test


def train_assign_direction_by_period(X_train, Y_train, X_test):
    """
    use xgboost to train and give directions        # TODO: modify comments 
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
    :param raw_data: the data of the original csv 
    :return the decision after each minute  # TODO: modify comments 
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
        direction = direction.append(direction_curr_period)
        print(f'Finish training period {i}')
    return direction


# TODO Add comments 
# TODO fix alignment issue 
def trade(raw_data, direction):
    """
    
    """
    direction = pd.Series(direction)
    trade_start_idx = raw_data.shape[0] - direction.shape[0]
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
    raw_data = load_data()
    direction = train_assign_direction(raw_data)
    trade(raw_data, direction)
