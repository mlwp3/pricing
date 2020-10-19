# -*- coding: utf-8 -*-

import pandas as pd

def load_file():
    data_train_file = r'../../Data/train.csv'
    data_train = pd.read_csv(data_train_file).set_index('IDpol').drop('RecordID', axis='columns')
    
    data_test_file = r'../../Data/test.csv'
    data_test = pd.read_csv(data_test_file).set_index('IDpol').drop('RecordID', axis='columns')
    
    data_train['VehAgeBand'] = data_train['VehAgeBand'].astype(str)
    data_test['VehAgeBand']  = data_test['VehAgeBand'].astype(str)
    
    return data_train, data_test
    
def clean_data(data_raw):
    
    #create dataframe to adjust
    data = data_raw.copy()

    #
    # Process Target Vars
    #

#    data = data.loc[data['ClaimNb']<=5]
#    
#    data['Claim_xs000k'] = 0
#    data['Claim_xs000k'] = data['Claim_xs000k'].where(data['ClaimAmount']==0,1)
#    data['Claim_xs100k'] = 0
#    data['Claim_xs100k'] = data['Claim_xs100k'].where(data['ClaimAmount']<100000,1)
#    data['Claim_xs500k'] = 0
#    data['Claim_xs500k'] = data['Claim_xs500k'].where(data['ClaimAmount']<500000,1)



    
    return data



if __name__ =='__main__':
    
    
    data_train_raw, data_test_raw = load_file()
    
    data_test = data_test_raw.copy()
    data_train = data_train_raw.copy()
    
    # apply cleaning steps [:limit rows]
    data_train = clean_data(data_train_raw)
    data_test = clean_data(data_test_raw)
