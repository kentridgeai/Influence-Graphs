# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 17:21:04 2025

@author: User
"""

import numpy as np 
import torchvision.transforms as transforms
import openml 
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
import pandas
import torch


NoneType = type(None)
class DataFrameImputer(TransformerMixin):
    
    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):
        # print(y)
        if type(y) == NoneType:
            self.fill = pandas.Series([X[c].value_counts().index[0]
                if X[c].dtype == 'category' else X[c].median() for c in X],
                index=X.columns)
        else:
            self.fill = y

        return self

    def transform(self, X, y=None):
        # print(self.fill)
        return X.fillna(self.fill),self.fill
    
def dataframe_to_torch(X,y):
    
    X = pandas.get_dummies(X)
    
    Tx = pandas.DataFrame.to_numpy(X)
    TX = np.zeros(Tx.shape)
    Ty,cats = pandas.factorize(y,sort=True)
    for i in range(Tx.shape[1]):
        if X.dtypes[i].name == 'category':
            # print('halle')
            Tx[:,i],cats = pandas.factorize(Tx[:,i],sort=True)
            TX[:,i] = Tx[:,i].astype('float32')
        else:
            TX[:,i] = Tx[:,i].astype('float32')
    
    # y = 
    X = torch.from_numpy(TX).float()
    y = torch.from_numpy(Ty)
    return X,y


    
def  load_data_and_generators(dataset_id,partition=0.8):

    task = openml.tasks.get_task(task_id=int(dataset_id), download_splits=False,download_data=False,download_qualities=False,download_features_meta_data=False)
    dataset = task.get_dataset()
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute,
    )
    
    
    if isinstance(y[1], bool):
        y = y.astype('bool')
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=1-partition,
        random_state=11, 
        stratify=y,
        shuffle=True,
    )
    train_column_nan_info = X_train.isna().all()
    test_column_nan_info = X_test.isna().all()
    only_nan_columns = [label for label, value in train_column_nan_info.items() if value]
    test_nan_columns = [label for label, value in test_column_nan_info.items() if value]
    only_nan_columns.extend(test_nan_columns)
    only_nan_columns = set(only_nan_columns)
    X_train.drop(only_nan_columns, axis='columns', inplace=True)
    X_test.drop(only_nan_columns, axis='columns', inplace=True)
    
    # X = X.cuda()  #train_dataset.train_data is a Tensor(input data)
    # y = y.cuda()
    X_train,fill = DataFrameImputer().fit_transform(X_train)
    X_test,__ = DataFrameImputer().fit_transform(X_test,y=fill)
    
    X_train,y_train = dataframe_to_torch(X_train, y_train)
    X_test,y_test = dataframe_to_torch(X_test, y_test)
    
    return X_train, y_train, X_test, y_test 


    # y_test = 1 - y_test