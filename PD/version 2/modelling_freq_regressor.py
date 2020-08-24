# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import make_classification

import matplotlib.pyplot as plt

import data_cleansing as dc
import data_utils as du


#%%########################################################################
#
# load the and process data
#

data_train_raw, data_test_raw = dc.load_file()

data_test = data_test_raw.copy()
data_train = data_train_raw.copy()


#%%########################################################################
#
# Frequency Projection
#

#%% set target
data_train['freq'] = data_train['ClaimNb']/data_train['Exposure']
data_test['freq'] = data_test['ClaimNb']/data_test['Exposure']
target_name='freq'

#drop alternative targets
drop_list = ['ClaimNb', 'Exposure', 'ClaimAmount', 'BonusMalus', 'severity']
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

#%% split factors and target

#sets used to train
X_train = data_train_resampled.drop(target_name, axis='columns')
y_train = data_train_resampled[target_name]

#sets used to predict
X_train_to_predict = data_train_encoded.drop(target_name, axis='columns')
X_test = data_test_encoded.drop(target_name, axis='columns')


#%%
#
# fit a random forest
#
if False:
    clf_rf = RandomForestRegressor(min_samples_leaf=5, 
                                    n_estimators=100,
                                    max_depth=50,
                                    max_features='auto',
                                    criterion='mse',    #mse or mae
                                    verbose=True,
                                    n_jobs=-1)
    clf_rf = clf_rf.fit(X_train.values, y_train.values)    

#%% predict values
y_train_predicted_rf = clf_rf.predict(X_train_to_predict.values)    
y_test_predicted_rf = clf_rf.predict(X_test.values)   

#%% add results back to dataset
data_train_raw['freq_predicted'] = y_train_predicted_rf
data_train_raw['ClaimNb_predicted'] = y_train_predicted_rf*data_train_raw['Exposure']
data_train_raw['ClaimNb_predicted'] = data_train_raw['ClaimNb_predicted']>0.55

data_test_raw['freq_predicted'] = y_test_predicted_rf
data_test_raw['ClaimNb_predicted'] = y_test_predicted_rf*data_test_raw['Exposure']
data_test_raw['ClaimNb_predicted'] = data_test_raw['ClaimNb_predicted']>0.55


#%% print resutls
print('Train Original :', len(data_train_raw), data_train_raw['ClaimNb'].sum())
print('Train Predicted:', len(data_train_raw), data_train_raw['ClaimNb_predicted'].sum())

print('Train Original :', len(data_test_raw), data_test_raw['ClaimNb'].sum())
print('Train Predicted:', len(data_test_raw), data_test_raw['ClaimNb_predicted'].sum())

print('Train Gini     :', du.gini(data_train_raw['ClaimNb'], data_train_raw['ClaimNb_predicted']))
print('Test Gini      :', du.gini(data_test_raw['ClaimNb'], data_test_raw['ClaimNb_predicted']))

#df_features = pd.DataFrame(clf_rf.feature_importances_, index=X_train.columns)


#%%########################################################################
#
# Output to file
#

data_train_raw.to_csv(r'data_train_output.csv')
data_test_raw.to_csv(r'data_test_output.csv')





