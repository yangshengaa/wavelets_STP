# sklearn package (use whatever I could to perform classification)
# xgboost, maybe? 

# load packages
import os 
import numpy as np 
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

# models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# constants 
curr_folder = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(curr_folder, 'preprocess/510050_1m_wavelets.csv')


def load_data():
    """ read in preprocess data, obtain X and Y"""
    print('Loading Data ...')
    tbl = pd.read_csv(file_path)
    Y = tbl.label.to_numpy()
    X = tbl.drop(columns=['label']).to_numpy()
    print('Finish Loading')
    return X, Y


if __name__ == '__main__':
    X, Y = load_data()
    X = PCA(n_components=10).fit_transform(X)
    # train test split 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    # logistic regression 
    lg = LogisticRegression(multi_class='multinomial')
    lg_fit = lg.fit(X_train, Y_train)
    print('train accuracy: {}'.format(lg.score(X_train, Y_train)))
    print('test  accuracy: {}'.format(lg.score(X_test, Y_test)))
    print(confusion_matrix(Y_train, lg.predict(X_train)))
    print(confusion_matrix(Y_test, lg.predict(X_test)))
    # confusion matrix 

    # tree 

    # boosting 
    
