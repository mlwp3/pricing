# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import r2_score,mean_squared_error

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

target_name='ClaimNb'

#drop alternative targets
drop_list = [ 'Exposure', 'ClaimAmount', 'BonusMalus', 'severity', 'Claim_xs000k', 'Claim_xs100k','Claim_xs500k']
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
print(data_train_encoded[target_name].value_counts())
print(data_train_resampled[target_name].value_counts())


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
# Run a full model
#

clf_rf = RandomForestRegressor(min_samples_leaf=5, 
                                n_estimators=30,
                                max_depth=50,
                                max_features='auto',
                                criterion='mse',    #mse or mae
                                verbose=True,
                                oob_score=True,
                                n_jobs=3)
#fit values
clf_rf = clf_rf.fit(X_train.values, y_train.values)    
#predict values
y_train_predicted_rf = clf_rf.predict(X_train_to_predict.values)    
y_test_predicted_rf = clf_rf.predict(X_test.values)   

#%%
#gini results
train_results_gini = du.gini(data_train[target_name].values, list(y_train_predicted_rf))
test_results_gini = du.gini(y_test.values, y_test_predicted_rf)
#mean squared error
train_results_mse = mean_squared_error(data_train[target_name].values, list(y_train_predicted_rf))
test_results_mse = mean_squared_error(y_test.values, y_test_predicted_rf)



print('Results: ',
      'Tr_G {:,.5f}'.format(train_results_gini),
      'Te_G {:,.5f}'.format(test_results_gini),
      'Tr_mse {:,.5f}'.format(train_results_mse),
      'Te_mse {:,.5f}'.format(test_results_mse))




#%%
#
# Parameter tuning
#




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
    clf_rf = RandomForestRegressor(min_samples_leaf=5, 
                                    n_estimators=estimator,
                                    max_depth=50,
                                    max_features='auto',
                                    criterion='mse',    #mse or mae
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

    print(value,
          'Tr_G {:,.5f}'.format(train_results_gini[value]),
          'Te_G {:,.5f}'.format(test_results_gini[value]))
     
    

#%%
#
# max depth
#

print('---- max_depth -----')
test_range = [5, 10, 20, 30, 60, 100]

train_results_gini = {}
test_results_gini = {}
train_results_auc = {}
test_results_auc = {}
for value in test_range:
    clf_rf = RandomForestRegressor(min_samples_leaf=5, 
                                    n_estimators=20,
                                    max_depth=value,
                                    max_features='auto',
                                    criterion='mse',    #mse or mae
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

    print(value,
          'Tr_G {:,.5f}'.format(train_results_gini[value]),
          'Te_G {:,.5f}'.format(test_results_gini[value]))
     

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
    clf_rf = RandomForestRegressor(min_samples_leaf=value, 
                                    n_estimators=20,
                                    max_depth=20,
                                    max_features='auto',
                                    criterion='mse',    #mse or mae
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

    print(value,
          'Tr_G {:,.5f}'.format(train_results_gini[value]),
          'Te_G {:,.5f}'.format(test_results_gini[value]))
     

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
    clf_rf = RandomForestRegressor(min_samples_leaf=5,
                                    min_samples_split=value,
                                    n_estimators=20,
                                    max_depth=20,
                                    max_features='auto',
                                    criterion='mse',    #mse or mae
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

    print(value,
          'Tr_G {:,.5f}'.format(train_results_gini[value]),
          'Te_G {:,.5f}'.format(test_results_gini[value]))
     

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
    clf_rf = RandomForestRegressor(min_samples_leaf=5,
                                    min_samples_split=5,
                                    n_estimators=20,
                                    max_depth=20,
                                    max_features=value,
                                    criterion='mse',    #mse or mae
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

    print(value,
          'Tr_G {:,.5f}'.format(train_results_gini[value]),
          'Te_G {:,.5f}'.format(test_results_gini[value]))
     




