# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. acquire
4. prepare_mvp
5. wrangle_wines_mvp
6. split
7. remove_outliers
'''

# =======================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientation START
# =======================================================================================================

'''
The purpose of this file is to create functions for both the acquire & preparation phase of the data
science pipeline or also known as 'wrangle' data...
'''

# =======================================================================================================
# Orientation END
# Orientation TO Imports
# Imports START
# =======================================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
import os
import env

# =======================================================================================================
# Imports END
# Imports TO acquire
# acquire START
# =======================================================================================================

def acquire():
    '''
    Obtains the vanilla version of both the red and white wine dataframe

    INPUT:
    NONE

    OUTPUT:
    red = pandas dataframe with red wine data
    white = pandas dataframe with white wine data
    '''
    red = pd.read_csv('https://query.data.world/s/k6viyg23e4usmgc2joiodhf2pvcvao?dws=00000')
    white = pd.read_csv('https://query.data.world/s/d5jg7efmkn3kq7cmrvvfkx2ww7epq7?dws=00000')
    return red, white

# =======================================================================================================
# acquire END
# acquire TO prepare_mvp
# prepare_mvp START
# =======================================================================================================

def prepare_mvp():
    '''
    Takes in the vanilla red and white wine dataframes and returns a cleaned version that is ready 
    for exploration and further analysis

    INPUT:
    NONE

    OUTPUT:
    wines = pandas dataframe with both red and white wine prepped for exploration
    '''
    red, white = acquire()
    white['wine_color'] = 'white'
    red['wine_color'] = 'red'
    wines = pd.concat([red, white], axis=0)
    return wines

# =======================================================================================================
# prepare_mvp END
# prepare_mvp TO wrangle_wines_mvp
# wrangle_wines_mvp START
# =======================================================================================================

def wrangle_wines_mvp():
    '''
    Function that acquires, prepares, and splits the wines dataframe for use as well as 
    creating a csv.

    INPUT:
    NONE

    OUTPUT:
    .csv = ONLY IF FILE NONEXISTANT
    wines = pandas dataframe with both red and white wine prepped for exploration
    '''
    if os.path.exists('wines.csv'):
        wines = pd.read_csv('wines.csv', index_col=0)
        train, validate, test = split(wines)
        return train, validate, test
    else:
        wines = prepare_mvp()
        wines.to_csv('wines.csv')
        train, validate, test = split(wines)
        return train, validate, test
    
# =======================================================================================================
# wrangle_zillow END
# wrangle_zillow TO split
# split START
# =======================================================================================================

def split(df):
    '''
    Takes a dataframe and splits the data into a train, validate and test datasets
    '''
    train_val, test = train_test_split(df, train_size=0.8, random_state=1349)
    train, validate = train_test_split(train_val, train_size=0.7, random_state=1349)
    print(f"train.shape:{train.shape}\nvalidate.shape:{validate.shape}\ntest.shape:{test.shape}")
    return train, validate, test


# =======================================================================================================
# split END
# split TO scale
# scale START
# =======================================================================================================

def scale(train, validate, test, cols, scaler):
    '''
    Takes in a train, validate, test and returns the dataframes,
    but scaled using the 'StandardScaler()'
    '''
    original_train = train.copy()
    original_validate = validate.copy()
    original_test = test.copy()
    scale_cols = cols
    scaler = scaler
    scaler.fit(original_train[scale_cols])
    original_train[scale_cols] = scaler.transform(original_train[scale_cols])
    scaler.fit(original_validate[scale_cols])
    original_validate[scale_cols] = scaler.transform(original_validate[scale_cols])
    scaler.fit(original_test[scale_cols])
    original_test[scale_cols] = scaler.transform(original_test[scale_cols])
    new_train = original_train
    new_validate = original_validate
    new_test = original_test
    return new_train, new_validate, new_test

# =======================================================================================================
# scale END
# scale TO sample_dataframe
# sample_dataframe START
# =======================================================================================================

def sample_dataframe(train, validate, test):
    '''
    Takes train, validate, test dataframes and reduces the shape to no more than 1000 rows by taking
    the percentage of 1000/len(train) then applying that to train, validate, test dataframes.

    INPUT:
    train = Split dataframe for training
    validate = Split dataframe for validation
    test = Split dataframe for testing

    OUTPUT:
    train_sample = Reduced size of original split dataframe of no more than 1000 rows
    validate_sample = Reduced size of original split dataframe of no more than 1000 rows
    test_sample = Reduced size of original split dataframe of no more than 1000 rows
    '''
    ratio = 1000/len(train)
    train_samples = int(ratio * len(train))
    validate_samples = int(ratio * len(validate))
    test_samples = int(ratio * len(test))
    train_sample = train.sample(train_samples)
    validate_sample = validate.sample(validate_samples)
    test_sample = test.sample(test_samples)
    return train_sample, validate_sample, test_sample

# =======================================================================================================
# sample_dataframe END
# sample_dataframe TO remove_outliers
# remove_outliers START
# =======================================================================================================

def remove_outliers(df, col_list, k=1.5):
    '''
    remove outliers from a dataframe based on a list of columns using the tukey method
    returns a single dataframe with outliers removed
    '''
    col_qs = {}
    for col in col_list:
        col_qs[col] = q1, q3 = df[col].quantile([0.25, 0.75])
    for col in col_list:
        iqr = col_qs[col][0.75] - col_qs[col][0.25]
        lower_fence = col_qs[col][0.25] - (k*iqr)
        upper_fence = col_qs[col][0.75] + (k*iqr)
        df = df[(df[col] > lower_fence) & (df[col] < upper_fence)]
    return df

# =======================================================================================================
# remove_outliers END
# =======================================================================================================