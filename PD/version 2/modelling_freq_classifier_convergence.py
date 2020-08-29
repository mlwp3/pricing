# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_curve, auc

from sklearn.datasets import make_classification

import matplotlib.pyplot as plt

import data_cleansing as dc
import data_utils as du


#%%########################################################################
#
# load the and process data
#

data_train_raw, data_test_raw = dc.load_file()

data_test_raw = dc.clean_data(data_test_raw)
data_train_raw = dc.clean_data(data_train_raw)

data_test = data_test_raw.copy()
data_train = data_train_raw.copy()

#%%########################################################################
#
# Frequency Projection
#

#%% set target
#data_train['freq'] = data_train['ClaimNb']/data_train['Exposure']
#data_test['freq'] = data_test['ClaimNb']/data_test['Exposure']
#target_name='freq'

target_name='Claim_xs000k'

#drop alternative targets
drop_list = ['ClaimNb', 'Exposure', 'ClaimAmount', 'BonusMalus', 'severity', 'Claim_xs100k','Claim_xs500k']
data_train.drop(drop_list,axis='columns', inplace=True)
data_test.drop(drop_list,axis='columns', inplace=True)

#%% lable encode factors
label_encode_factors = ['Area',
                        'VehPower',
                        'VehBrand',
                        'VehGas',
                        #'Density',
                        'Region',
                        'DrivAgeBand',
                        'DensityBand',
                        'VehAgeBand']
data_train_encoded, encoders = du.preprocess_labelencode(data_train, label_encode_factors)
data_test_encoded = du.preprocess_labelencode_apply(encoders, data_test, label_encode_factors)


#%% oversample the claim ovservations to create a balanced data set
data_train_resampled = du.oversample_training_set(data_train_encoded, target_name)
#data_train_resampled = data_train_encoded.copy()

#%% split factors and target

#sets used to train (resampled sets)
X_train = data_train_resampled.drop(target_name, axis='columns')
y_train = data_train_resampled[target_name]
y_test = data_test_encoded[target_name]

#sets used to predict
X_train_to_predict = data_train_encoded.drop(target_name, axis='columns')
X_test = data_test_encoded.drop(target_name, axis='columns')


#%%
#
# n_estimators - test convergence
#


n_estimators = [5, 10, 20, 30, 60, 100, 200]

train_results_gini = {}
test_results_gini = {}
train_results_auc = {}
test_results_auc = {}
for estimator in n_estimators:
    clf_rf = RandomForestClassifier(min_samples_leaf=5, 
                                    n_estimators=estimator,
                                    max_depth=50,
                                    max_features='auto',
                                    criterion='gini',    #mse or mae
                                    verbose=False,
                                    #oob_score=True,
                                    n_jobs=3)
    #fit values
    clf_rf = clf_rf.fit(X_train.values, y_train.values)    
    #predict values
    y_train_predicted_rf = clf_rf.predict(X_train_to_predict.values)    
    y_test_predicted_rf = clf_rf.predict(X_test.values)   

    #gini results
    train_results_gini[estimator] = du.gini(data_train[target_name].values, list(y_train_predicted_rf))
    test_results_gini[estimator] = du.gini(y_test.values, y_test_predicted_rf)
    #auc results
    fp_rate_train, to_rate_train, thresholds_train = roc_curve(data_train[target_name].values, list(y_train_predicted_rf))
    fp_rate_test, to_rate_test, thresholds_test = roc_curve(y_test.values, y_test_predicted_rf)
    #store results
    roc_auc_train = auc(fp_rate_train, to_rate_train)
    roc_auc_test = auc(fp_rate_test, to_rate_test)
    train_results_auc[estimator] = roc_auc_train
    test_results_auc[estimator] = roc_auc_test

    print(estimator,
          'Tr_G {:,.5f}'.format(train_results_gini[estimator]),
          'Te_G {:,.5f}'.format(test_results_gini[estimator]),
          'Tr_A {:,.5f}'.format(train_results_auc[estimator]),
          'Te_A {:,.5f}'.format(test_results_auc[estimator]))
    

#%%
#
# max depth
#


test_range = [5, 10, 20, 30, 60, 100]

train_results_gini = {}
test_results_gini = {}
train_results_auc = {}
test_results_auc = {}
for value in test_range:
    clf_rf = RandomForestClassifier(min_samples_leaf=5, 
                                    n_estimators=20,
                                    max_depth=value,
                                    max_features='auto',
                                    criterion='gini',    #mse or mae
                                    verbose=False,
                                    #oob_score=True,
                                    n_jobs=3)
    #fit values
    clf_rf = clf_rf.fit(X_train.values, y_train.values)    
    #predict values
    y_train_predicted_rf = clf_rf.predict(X_train_to_predict.values)    
    y_test_predicted_rf = clf_rf.predict(X_test.values)   

    #gini results
    train_results_gini[value] = du.gini(data_train[target_name].values, list(y_train_predicted_rf))
    test_results_gini[value] = du.gini(y_test.values, y_test_predicted_rf)
    #auc results
    fp_rate_train, to_rate_train, thresholds_train = roc_curve(data_train[target_name].values, list(y_train_predicted_rf))
    fp_rate_test, to_rate_test, thresholds_test = roc_curve(y_test.values, y_test_predicted_rf)
    #store results
    roc_auc_train = auc(fp_rate_train, to_rate_train)
    roc_auc_test = auc(fp_rate_test, to_rate_test)
    train_results_auc[value] = roc_auc_train
    test_results_auc[value] = roc_auc_test

    print(value,
          'Tr_G {:,.5f}'.format(train_results_gini[value]),
          'Te_G {:,.5f}'.format(test_results_gini[value]),
          'Tr_A {:,.5f}'.format(train_results_auc[value]),
          'Te_A {:,.5f}'.format(test_results_auc[value]))
    

#%%
#
# min_leaf_samples depth
#


test_range = [5, 50, 100, 200, 500, 1000]

train_results_gini = {}
test_results_gini = {}
train_results_auc = {}
test_results_auc = {}
for value in test_range:
    clf_rf = RandomForestClassifier(min_samples_leaf=value, 
                                    n_estimators=20,
                                    max_depth=20,
                                    max_features='auto',
                                    criterion='gini',    #mse or mae
                                    verbose=False,
                                    #oob_score=True,
                                    n_jobs=3)
    #fit values
    clf_rf = clf_rf.fit(X_train.values, y_train.values)    
    #predict values
    y_train_predicted_rf = clf_rf.predict(X_train_to_predict.values)    
    y_test_predicted_rf = clf_rf.predict(X_test.values)   

    #gini results
    train_results_gini[value] = du.gini(data_train[target_name].values, list(y_train_predicted_rf))
    test_results_gini[value] = du.gini(y_test.values, y_test_predicted_rf)
    #auc results
    fp_rate_train, to_rate_train, thresholds_train = roc_curve(data_train[target_name].values, list(y_train_predicted_rf))
    fp_rate_test, to_rate_test, thresholds_test = roc_curve(y_test.values, y_test_predicted_rf)
    #store results
    roc_auc_train = auc(fp_rate_train, to_rate_train)
    roc_auc_test = auc(fp_rate_test, to_rate_test)
    train_results_auc[value] = roc_auc_train
    test_results_auc[value] = roc_auc_test

    print(value,
          'Tr_G {:,.5f}'.format(train_results_gini[value]),
          'Te_G {:,.5f}'.format(test_results_gini[value]),
          'Tr_A {:,.5f}'.format(train_results_auc[value]),
          'Te_A {:,.5f}'.format(test_results_auc[value]))

#%%
#
# min_samples_split
#


test_range = [5, 10, 20, 50,80,100]

train_results_gini = {}
test_results_gini = {}
train_results_auc = {}
test_results_auc = {}
for value in test_range:
    clf_rf = RandomForestClassifier(min_samples_leaf=5,
                                    min_samples_split=value,
                                    n_estimators=20,
                                    max_depth=20,
                                    max_features='auto',
                                    criterion='gini',    #mse or mae
                                    verbose=False,
                                    #oob_score=True,
                                    n_jobs=3)
    #fit values
    clf_rf = clf_rf.fit(X_train.values, y_train.values)    
    #predict values
    y_train_predicted_rf = clf_rf.predict(X_train_to_predict.values)    
    y_test_predicted_rf = clf_rf.predict(X_test.values)   

    #gini results
    train_results_gini[value] = du.gini(data_train[target_name].values, list(y_train_predicted_rf))
    test_results_gini[value] = du.gini(y_test.values, y_test_predicted_rf)
    #auc results
    fp_rate_train, to_rate_train, thresholds_train = roc_curve(data_train[target_name].values, list(y_train_predicted_rf))
    fp_rate_test, to_rate_test, thresholds_test = roc_curve(y_test.values, y_test_predicted_rf)
    #store results
    roc_auc_train = auc(fp_rate_train, to_rate_train)
    roc_auc_test = auc(fp_rate_test, to_rate_test)
    train_results_auc[value] = roc_auc_train
    test_results_auc[value] = roc_auc_test

    print(value,
          'Tr_G {:,.5f}'.format(train_results_gini[value]),
          'Te_G {:,.5f}'.format(test_results_gini[value]),
          'Tr_A {:,.5f}'.format(train_results_auc[value]),
          'Te_A {:,.5f}'.format(test_results_auc[value]))

#%%
#
# max features
#


test_range = [2, 3, 4, 5,6,7,8,9,10]

train_results_gini = {}
test_results_gini = {}
train_results_auc = {}
test_results_auc = {}
for value in test_range:
    clf_rf = RandomForestClassifier(min_samples_leaf=5,
                                    min_samples_split=5,
                                    n_estimators=20,
                                    max_depth=20,
                                    max_features=value,
                                    criterion='gini',    #mse or mae
                                    verbose=False,
                                    #oob_score=True,
                                    n_jobs=3)
    #fit values
    clf_rf = clf_rf.fit(X_train.values, y_train.values)    
    #predict values
    y_train_predicted_rf = clf_rf.predict(X_train_to_predict.values)    
    y_test_predicted_rf = clf_rf.predict(X_test.values)   

    #gini results
    train_results_gini[value] = du.gini(data_train[target_name].values, list(y_train_predicted_rf))
    test_results_gini[value] = du.gini(y_test.values, y_test_predicted_rf)
    #auc results
    fp_rate_train, to_rate_train, thresholds_train = roc_curve(data_train[target_name].values, list(y_train_predicted_rf))
    fp_rate_test, to_rate_test, thresholds_test = roc_curve(y_test.values, y_test_predicted_rf)
    #store results
    roc_auc_train = auc(fp_rate_train, to_rate_train)
    roc_auc_test = auc(fp_rate_test, to_rate_test)
    train_results_auc[value] = roc_auc_train
    test_results_auc[value] = roc_auc_test

    print(value,
          'Tr_G {:,.5f}'.format(train_results_gini[value]),
          'Te_G {:,.5f}'.format(test_results_gini[value]),
          'Tr_A {:,.5f}'.format(train_results_auc[value]),
          'Te_A {:,.5f}'.format(test_results_auc[value]))




