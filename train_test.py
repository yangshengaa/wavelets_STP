"""
File name: train_test.py
author: Sheng Yang
Date Created: 2021/04/12

This file compares performances of different ML models on 
predicting the stock trend using wavelet coefficients 

TODO: change to reading parquet 

"""

# load preprocessing
import os
import multiprocessing as mp
import pandas as pd
from functools import partial

# preprocess 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

# models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# load plots 
import matplotlib.pyplot as plt 

# constants 
curr_folder = os.path.dirname(os.path.abspath(__file__))
report_path = 'report'
acc_report_file_path = 'report/accuracy.txt'


def load_preprocess_data():
    """ read in preprocess data, obtain X and Y"""
    print('Loading Data ...')
    tbl = pd.DataFrame()
    files_to_read = os.listdir('preprocess')
    files_to_read.sort()
    files_to_read = [os.path.join('preprocess', f) for f in files_to_read]

    # mp pool to read
    tbl_list = []
    read_df = partial(pd.read_csv, header=None)
    with mp.Pool() as pool:
        tbl_list = pool.map(read_df, files_to_read)
    tbl = pd.concat(tbl_list).to_numpy()
    Y = tbl[:, -1]
    X = tbl[:, :-1]
    print('Finish Loading')
    return X, Y


def write_report(model_type, train_acc, test_acc):
    """ log accuracy of each model to a file """
    with open(acc_report_file_path, 'a') as f:
        f.write(f'{model_type} \n')
        f.write(f'Train Score: {train_acc}\n')
        f.write(f'Test  Score: {test_acc}\n')
        f.write('\n')


# -------- models to use -------------


def train_test_lg(X_train, X_test, Y_train, Y_test):
    """
    logistic regression
    """
    # training
    lg = LogisticRegression(multi_class='multinomial')
    lg_fit = lg.fit(X_train, Y_train)
    # metrics
    lg_train_acc = lg.score(X_train, Y_train)
    lg_test_acc = lg.score(X_test, Y_test)
    # plot and save matrix
    plot_confusion_matrix(lg, X_test, Y_test, normalize='true', cmap='Blues')
    plt.savefig(os.path.join(report_path, 'lg_test_confusion_matrix.png'))
    # record metrics
    write_report('Logistic Regression', lg_train_acc, lg_test_acc)


def train_test_boosting(X_train, X_test, Y_train, Y_test):
    """
    gradient boosting classification
    """
    # training
    boosting_clf = GradientBoostingClassifier().fit(X_train, Y_train)
    # obtaining metrics
    boosting_train_acc = boosting_clf.score(X_train, Y_train)
    boosting_test_acc = boosting_clf.score(X_test, Y_test)
    # record plots
    plot_confusion_matrix(boosting_clf, X_test, Y_test, normalize='true', cmap='Blues')
    plt.savefig(os.path.join(report_path, 'boosting_test_confusion_matrix.png'))
    # record metrics
    write_report('Gradient Boosting', boosting_train_acc, boosting_test_acc)


def train_test_svc(X_train, X_test, Y_train, Y_test):
    """
    linear SVM classification 
    """
    # training 
    svm_clf = LinearSVC().fit(X_train, Y_train)
    # obtaining metrics 
    svm_train_acc = svm_clf.score(X_train, Y_train)
    svm_test_acc = svm_clf.score(X_test, Y_test)
    # record plots 
    plot_confusion_matrix(svm_clf, X_test, Y_test, normalize='true', cmap='Blues')
    plt.savefig(os.path.join(report_path, 'svm_test_confusion_matrix.png'))
    # record metrics 
    write_report('Linear SVM', svm_train_acc, svm_test_acc)


def train_test_xgb(X_train, X_test, Y_train, Y_test):
    """
    XGBoost classification
    """
    # training
    xgb_clf = XGBClassifier(
        objective='multi:softmax', 
        use_label_encoder=False
        ).fit(X_train, Y_train, 
              eval_metric='merror')
    Y_train_predict = xgb_clf.predict(X_train) 
    Y_test_predict = xgb_clf.predict(X_test) 
    xgb_train_acc = (Y_train_predict == Y_train).mean()
    xgb_test_acc = (Y_test_predict == Y_test).mean()
    # record plots
    plot_confusion_matrix(xgb_clf, X_test, Y_test,
                          normalize='true', cmap='Blues')
    plt.savefig(os.path.join(report_path, 'xgb_test_confusion_matrix.png'))
    # record metrics
    write_report('XGBoost Classifier', xgb_train_acc, xgb_test_acc)


"""
Everything below is for testing purposes. No need to run this script.
"""
if __name__ == '__main__':
    # reading in the data
    X, Y = load_preprocess_data()

    # sequential split
    X_train_complete, X_test_complete, Y_train, Y_test = \
        train_test_split(X, Y, shuffle=False)

    # PCA
    pca = PCA(n_components=20)
    pca.fit(X_train_complete)
    X_train = pca.transform(X_train_complete)
    X_test = pca.transform(X_test_complete)

    print('Start Training')
    train_test_xgb(X_train, X_test, Y_train, Y_test)
    print('End Training')

    # the followings are for future build: some sort of running file in main instead of jupyter

    # dump file content
    # with open(os.path.join(report_path, 'accuracy.txt'), 'w') as f:
    #     f.write('')
    
    # # initialize processes
    # p_lg = mp.Process(target=train_test_lg, args=(X_train, X_test, Y_train, Y_test))
    # p_boosting = mp.Process(target=train_test_boosting, args=(X_train, X_test, Y_train, Y_test))
    # p_list = [p_lg, p_boosting]
    
    # # start training
    # print('Start Training and Testing')
    # for p in p_list:
    #     p.start()
    # for p in p_list:
    #     p.join()
    # print('Finish Training and Testing')
