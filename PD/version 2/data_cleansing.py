# -*- coding: utf-8 -*-

import pandas as pd

def load_file(data_filename):
    return pd.read_csv(data_filename)
    


def clean_data_strict(data_raw):
    
    #create dataframe to adjust
    data = data_raw.copy()

    #
    # Process Target Vars
    #

    # ClaimNb_binary
    # create binary claim flag to use as categorical variable (remove multi claim policies)
    data['ClaimNb_Binary'] = 0
    data['ClaimNb_Binary'] = data['ClaimNb_Binary'].where(data['ClaimNb']==0,1)
    
    # Freq (claims per year)
    # adjust to definition from adjusted claim number
    data['Freq'] = data['ClaimNb_Binary']/data['Exposure']
    
    
    #
    # Process Factors
    #
    
    # Exposure
    # filter on exposure of more than 10% of year, and remove exposures>1year
    data = data.loc[(data['Exposure']>= 0.1) & (data['Exposure']<=1)]
    
    # Area
    # convert to category
    data['Area'] = data['Area'].astype('category')
    
    #VehPower
    map_vehpower = {1:'L',
                    2:'L',
                    3:'L',
                    4:'L',
                    5:'L',
                    6:'M',
                    7:'M',
                    8:'M',
                    9:'M',
                    10:'M',
                    11:'H',
                    12:'H',
                    13:'H',
                    14:'H',
                    15:'H',
                    }
    data['VehPower'].replace(map_vehpower, inplace=True)

    # BonusMalus
    bins = [49, 60, 80, 100, 200, 350]
    data['BonusMalus'] = pd.cut(data['BonusMalus'], bins)
    
    # VehBrand
    #nothing

    # VehGas
    #nothing

    # Region
    #nothing

    # DrivAgeBand
    # drop young drivers as there are very few observations
    data = data.loc[data['DrivAgeBand']!='18, 25']

    
    # DensityBand
    #nothing

    # VehAgeBand
    #nothing

    category_list = ['ClaimNb_Binary','VehPower','BonusMalus','DrivAgeBand','DensityBand','VehAgeBand','Region','VehGas','VehBrand']
    for cat in category_list:
        data[cat] = data[cat].astype('category')

    
    return data


def clean_data_minimal(data_raw):
    
    #create dataframe to adjust
    data = data_raw.copy()

    #
    # Process Target Vars
    #

    # ClaimNb_binary
    # create binary claim flag to use as categorical variable (remove multi claim policies)
    data['ClaimNb_Binary'] = 0
    data['ClaimNb_Binary'] = data['ClaimNb_Binary'].where(data['ClaimNb']==0,1)
    
    # Freq (claims per year)
    # adjust to definition from adjusted claim number
    data['Freq'] = data['ClaimNb_Binary']/data['Exposure']
    

    #
    # Apply filters to the data
    #

    # Exposure
    # filter on exposure of more than 10% of year, and remove exposures>1year
    data = data.loc[(data['Exposure']>= 0.1) & (data['Exposure']<=1)]

    # drop young drivers as there are very few observations and this is a high risk category
    data = data.loc[data['DrivAgeBand']!='18, 25']
    
    #
    # Process Factors
    #
    
    # Area
    # convert to category
    #nothing
    
    #VehPower
    #nothing

    # BonusMalus
    #nothing
    
    # VehBrand
    data['VehBrand'] = data['VehBrand'].apply(lambda x: x.replace('B', '')).astype(int)

    # VehGas
    #nothing

    # Region
    data['Region'] = data['Region'].apply(lambda x: x.replace('R', '')).astype(int)

    # DrivAgeBand
    #nothing
    
    # DensityBand
    #nothing

    # VehAgeBand
    data['VehAgeBand'] = data['VehAgeBand'].apply(lambda x: x.replace('+', '')).astype(int)

    category_list = ['ClaimNb_Binary','VehPower','BonusMalus','DrivAgeBand','DensityBand','VehAgeBand','Region','VehGas','VehBrand']
    for cat in category_list:
        data[cat] = data[cat].astype('category')
    
    return data



if __name__ =='__main__':
    
    
    data_train_file = r'../final/Data/train_new_final.csv'
    data_train_raw = load_file(data_train_file)
    
    data_test_file = r'../final/Data/test_new_final.csv'
    data_test_raw = load_file(data_test_file)
    
    # apply cleaning steps [:limit rows]
    data_train = clean_data_minimal(data_train_raw)
    data_test = clean_data_minimal(data_test_raw)
