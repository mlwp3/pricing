# -*- coding: utf-8 -*-

import pandas as pd

def load_file():
    data_train_file = r'../../final/Data/train_new_final.csv'
    data_train = pd.read_csv(data_train_file).set_index('RecordID')
    
    data_test_file = r'../../final/Data/test_new_final.csv'
    data_test = pd.read_csv(data_test_file).set_index('RecordID')
    
    return data_train, data_test
    
def clean_data_minimal(data_raw):
    
    #create dataframe to adjust
    data = data_raw.copy()

    #
    # Process Target Vars
    #

    #nothing

    #
    # Apply filters to the data
    #

    # Exposure
    #nothing

    
    #
    # Process Factors
    #
    
    # Area
    #nothing
    
    # VehPower
    #nothing

    # BonusMalus
    #nothing
    
    # VehBrand
    #nothing

    # VehGas
    #nothing

    # Region
    #nothing

    # DrivAgeBand
    #nothing
    
    # DensityBand
    #nothing

    # VehAgeBand
    #nothing

    
    return data



if __name__ =='__main__':
    
    
    data_train_raw, data_test_raw = load_file()
    
    data_test = data_test_raw.copy()
    data_train = data_train_raw.copy()
    
    # apply cleaning steps [:limit rows]
    data_train = clean_data_minimal(data_train_raw)
    data_test = clean_data_minimal(data_test_raw)
