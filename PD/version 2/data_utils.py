# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import numpy as np
import matplotlib.pyplot as plt

def preprocess_onehot(data):
    
    #fit one hot encoder to categorical data (all columns)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(data)
    data_out_array = enc.transform(data).toarray()
    column_names = enc.get_feature_names(data.columns)
    
    df_out = pd.DataFrame(data_out_array, columns=column_names)
    
    return df_out

def preprocess_labelencode(data, factor_list):
    ''' fit a label encoder and encode the data'''
    data_out = data.copy()
    encoders = {}
    for factor in factor_list:
        encoders[factor] = LabelEncoder()
        encoders[factor].fit(data[factor].values)
        data_out[factor] = encoders[factor].transform(data[factor])
    
    return data_out, encoders

def preprocess_labelencode_apply(encoders, data, factor_list):
    ''' apply prefit encoders to another dataset'''
    data_out = data.copy()
    for factor in factor_list:
        data_out[factor] = encoders[factor].transform(data[factor])
    
    return data_out
    

def oversample_training_set(data, target_name):
    data_training = data.copy()
    
    data_training_1 = pd.DataFrame(data_training.loc[data_training[target_name]>=1])
    data_training_0 = pd.DataFrame(data_training.loc[data_training[target_name]==0])

    difference = len(data_training_1) - len(data_training_0)
    
    if difference >0:
        #where len 1 is greater
        data_training_0 = data_training_0.sample(len(data_training_1), replace=True)
    else:
        data_training_1 = data_training_1.sample(len(data_training_0), replace=True)

    print('Target 0: ', len(data_training_0))
    print('Target 1: ', len(data_training_1))
    
    data_training_out = pd.concat([data_training_1, data_training_0])
    
    return data_training_out
    

def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)


def plot_factors(data_test, target_name, target_name_predicted, data_train=None):
    if data_train is not None:
        data_train_ref = data_train.copy()
    data_to_plot = data_test.copy()
    number_factors = len(data_to_plot.columns)
    
    fig, axs = plt.subplots(number_factors,1, figsize=(10,number_factors*10))
    
    for ax_index in range(0,len(axs)):
        factor = data_to_plot.columns[ax_index]
        print(factor)
        if factor in [target_name, target_name_predicted]:
            #nothing
            print('--target')
        else:
            data_to_plot[[target_name_predicted, factor]].groupby(factor).agg(np.mean).plot(kind='bar', ax=axs[ax_index])
            data_to_plot[[target_name, factor]].groupby(factor).agg(np.mean).plot(kind='line', color='orange',ax=axs[ax_index])
            axs[ax_index].legend(['Actual', 'Predicted'])
            if data_train is not None:
                data_train_ref[[target_name_predicted, factor]].groupby(factor).agg(np.mean).plot(kind='line', color='green',ax=axs[ax_index])
                #o, g, b
                axs[ax_index].legend(['Test_Actual', 'Train_Actual', 'Predicted'])
    return fig
