# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import matplotlib.pyplot as plt

import data_cleansing as dc
import data_utils as du


#%%########################################################################
#
# load the and process data
#

data_train_file = r'../final/Data/train_new_final.csv'
data_train_raw = dc.load_file(data_train_file)

data_test_file = r'../final/Data/test_new_final.csv'
data_test_raw = dc.load_file(data_test_file)

# apply cleaning steps [:limit rows]
data_train = dc.clean_data_strict(data_train_raw)
data_test = dc.clean_data_strict(data_test_raw)

#drop alternative targets
drop_list = ['ClaimNb','Freq', 'Exposure', 'ClaimAmount']
data_train.drop(drop_list,axis='columns', inplace=True)
data_test.drop(drop_list,axis='columns', inplace=True)

#oversample the claim ovservations to create a balanced data set
target_name='ClaimNb_Binary'
data_train_resampled = du.oversample_training_set(data_train, target_name)

#%%
#
# preprocessing and encoding
#

X_train = du.preprocess_onehot(data_train_resampled.drop(target_name, axis='columns'))
y_train=data_train_resampled[target_name]

X_test = du.preprocess_onehot(data_test.drop(target_name, axis='columns'))
y_test=data_test[target_name]

# check one-hot encoding gives the same columns
X_train.columns == X_test.columns


#%%########################################################################
#
# fit a random forest
#

clf_rf = RandomForestClassifier(min_samples_leaf=3, 
                                n_estimators=100,
                                criterion='gini',
                                verbose=True,
                                n_jobs=-1)
clf_rf = clf_rf.fit(X_train.values, y_train.values)    

y_train_predicted_rf = clf_rf.predict(X_train.values)    
y_test_predicted_rf = clf_rf.predict(X_test.values)   

y_train_predicted_rf_prob = clf_rf.predict_proba(X_train.values)    
y_test_predicted_rf_prob = clf_rf.predict_proba(X_test.values)   

y_zero_test = [0]*len(y_test)

#%% print resutls
print('Train Original :', len(y_train), np.sum(list(y_train.values)))
print('Train Predicted:',len(y_train_predicted_rf), np.sum(y_train_predicted_rf))
print('Test Original  :', len(y_test), np.sum(list(y_test.values)))
print('Test Predicted :',len(y_test_predicted_rf), np.sum(y_test_predicted_rf))

print('Train Gini     :', du.gini(y_train, y_train_predicted_rf))
print('Test Gini      :', du.gini(y_test, y_test_predicted_rf))
print('Zeros Gini     :', du.gini(y_test, y_zero_test))

df_features = pd.DataFrame(clf_rf.feature_importances_, index=X_train.columns)



#%%########################################################################
#
# Calc lift curve (gains chart)
#


import scikitplot as skplt

skplt.metrics.plot_cumulative_gain(y_test, y_test_predicted_rf_prob)
skplt.metrics.plot_cumulative_gain(y_train, y_train_predicted_rf_prob)









