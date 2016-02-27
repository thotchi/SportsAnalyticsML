# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 21:15:15 2015

@author: dave.chi
"""

# Load libraries
    import pandas as pd
    from pandas import DataFrame
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import json
    from sklearn import linear_model
    from sklearn.metrics import confusion_matrix
    
# Set directory
    path_wd = r'C:\Users\dave.chi\DataScience\BRFS_data'
    os.chdir(path_wd)

# Load file in Numpy
# Load into Pandas instead, do some manipulation, save as np.ndarray
# brfs_data = np.genfromtxt("data_continuous.csv",delimiter=",",skip_header=1)

# Load file in Pandas
    data_table  = pd.read_csv('data_continuous.csv')
    b = data_table.describe
    data_table[:10]
    brfs_df = pd.DataFrame(data_table)

    brfs_nan = brfs_df.copy()
    id(brfs_nan)
    id(brfs_df)
    
    brfs_df['GENHLTH']
    brfs_df[brfs_nan['X_MINACT1']>100]
    brfs_df.ix[brfs_nan['X_MINACT1']>100, :3]
    
    brfs_df.min(axis=0)
    brfs_df.max(axis=0)

# replace answers representing unknown/refused with NULL

# replace 77,88,99 in HLTH columns
# Get column numbers of columns

    col_99 = (4,5,6,7,8,10,11,12)
    col_999 = (9,9)
    col_9999 = (16,17,21,39,40)
    col_99900 = (24,25,28,29,32)
        
    for col in col_99:
        #print(col)
        brfs_nan.ix[:,col] = brfs_nan.ix[:,col].replace(77,np.nan)
        brfs_nan.ix[:,col] = brfs_nan.ix[:,col].replace(88,np.nan)
        brfs_nan.ix[:,col] = brfs_nan.ix[:,col].replace(99,np.nan)
    
    for col in col_999:
        #print(col)
        brfs_nan.ix[:,col] = brfs_nan.ix[:,col].replace(777,np.nan)
        brfs_nan.ix[:,col] = brfs_nan.ix[:,col].replace(888,np.nan)
        brfs_nan.ix[:,col] = brfs_nan.ix[:,col].replace(999,np.nan)
    
    for col in col_9999:
        #print(col)
        brfs_nan.ix[:,col] = brfs_nan.ix[:,col].replace(9900,np.nan)
        brfs_nan.ix[:,col] = brfs_nan.ix[:,col].replace(9999,np.nan)
        
    for col in col_99900:
        #print(col)
        brfs_nan.ix[:,col] = brfs_nan.ix[:,col].replace(99900,np.nan)
        brfs_nan.ix[:,col] = brfs_nan.ix[:,col].replace(99000,np.nan)

# Get statistics for all columns
    c=brfs_nan.describe()
    count = pd.DataFrame(np.arange(41))
    std_val=c[2:3].T
    med_val=c[5:6].T
    max_val=c[7:8].T
    plus_6sigma = pd.DataFrame(med_val['50%'] + 6*std_val['std'])
    plus_6sigma.columns = ['plus_6sigma']
    max_6simga_delta = pd.DataFrame(max_val['max']-plus_6sigma['plus_6sigma'])
    max_6simga_delta.columns = ['max_6simga_delta']
    
    DataFrame(std_val.index)
    
    stat = pd.concat([std_val, med_val, max_val, plus_6sigma, max_6simga_delta], axis=1)

# Filter out +6 sigma events
    stat.ix[:1,3:4]
    row_count = len(brfs_nan.index)
    col_count = len(brfs_nan.columns)
    
    brfs_filtered = brfs_nan.copy()
    
    brfs_filtered = brfs_filtered[np.abs(brfs_nan-brfs_nan.mean())<=(6*brfs_nan.std())]
    d=brfs_filtered.describe()

# Normalize columns to Z-score
    brfs_normalized = DataFrame()
        
    cols = list(brfs_filtered.columns)    
    brfs_normalized[cols]
    
    for col in cols:
        col_zscore = col + '_zscore'
        brfs_normalized[col_zscore] = (brfs_filtered[col] - brfs_filtered[col].mean())/brfs_filtered[col].std(ddof=0)

    GENHLTH_avg = brfs_filtered['GENHLTH'].mean()
    GENHLTH_sd = brfs_filtered['GENHLTH'].std(ddof=0)
    
    brfs_zero = brfs_normalized.copy()
    brfs_zero = brfs_normalized.fillna(0)

# Drop columns that don't have enough non-null values
    brfs_full_col = brfs_normalized.copy()
    CountNonNull = brfs_normalized.count(axis=0, level=None, numeric_only=True)
    for col in cols:
        if CountNonNull[col+'_zscore'] < 30000:
            brfs_full_col = brfs_full_col.drop(col+'_zscore', 1)
    brfs_zero = brfs_full_col.fillna(0)
    
# Create NumPy Arrays from pandas DataFrame
    fit_column = 'GENHLTH_zscore'
    factor_df = brfs_zero.drop(fit_column, 1)
    len(factor_df.columns)
    len(brfs_zero.columns)
    factor_array = factor_df.values    
    fit_array = brfs_zero[fit_column].values
    len(brfs_zero.columns)
    len(brfs_normalized.columns)
    
# Create Training and Test data sets
    training_count = 90000
    training_set = factor_array[:training_count]
    training_labels = fit_array[:training_count]
    test_set = factor_array[training_count:]
    test_labels = fit_array[training_count:]

# Test SGD regression model
    clf = linear_model.SGDRegressor()        
    clf.fit(training_set, training_labels)
    SGD_pred = clf.predict(test_set)

# Convert prediction back to GENHLTH score
    SGD_GENHLTH_pred = np.round(SGD_pred * GENHLTH_sd + GENHLTH_avg,0)
    GENHLTH_actual = np.round(test_labels * GENHLTH_sd + GENHLTH_avg,0)

    d = {'GENHLTH_actual' : GENHLTH_actual, 'SGD_GENHLTH_pred' : SGD_GENHLTH_pred}
    DataFrame(d)
# Confusion table
    SGD_Confusion_Matrix = DataFrame(confusion_matrix(SGD_GENHLTH_pred,GENHLTH_actual), index=[1,2,3,4,5], columns=[1,2,3,4,5])
    print(SGD_Confusion_Matrix)        
    col = np.arange(len(SGD_Confusion_Matrix.columns))
    true_sum = 0
    for c in col:
        print(c)
        true_sum = true_sum + SGD_Confusion_Matrix.iat[c,c]
    agreement_pct = true_sum /  SGD_Confusion_Matrix.sum().sum()
    print(agreement_pct)

# Test Ordinary Least Squares regression model
    clf = linear_model.LinearRegression()
    clf.fit(training_set, training_labels)
    SGD_pred = clf.predict(test_set)
# Convert prediction back to GENHLTH score
    SGD_GENHLTH_pred = np.round(SGD_pred * GENHLTH_sd + GENHLTH_avg,0)
    GENHLTH_actual = np.round(test_labels * GENHLTH_sd + GENHLTH_avg,0)
# Confustion table
    SGD_Confusion_Matrix = DataFrame(confusion_matrix(SGD_GENHLTH_pred,GENHLTH_actual), index=[1,2,3,4,5], columns=[1,2,3,4,5])
    print(SGD_Confusion_Matrix)        
    col = np.arange(len(SGD_Confusion_Matrix.columns))
    true_sum = 0
    for c in col:
        print(c)
        true_sum = true_sum + SGD_Confusion_Matrix.iat[c,c]
    agreement_pct = true_sum /  SGD_Confusion_Matrix.sum().sum()
    print(agreement_pct)
    
# Test Ridge Regression model
    clf = linear_model.Ridge (alpha = 1)
    clf.fit(training_set, training_labels)
    SGD_pred = clf.predict(test_set)
# Convert prediction back to GENHLTH score
    SGD_GENHLTH_pred = np.round(SGD_pred * GENHLTH_sd + GENHLTH_avg,0)
    GENHLTH_actual = np.round(test_labels * GENHLTH_sd + GENHLTH_avg,0)
# Confusion table
    SGD_Confusion_Matrix = DataFrame(confusion_matrix(SGD_GENHLTH_pred,GENHLTH_actual), index=[1,2,3,4,5], columns=[1,2,3,4,5])
    print(SGD_Confusion_Matrix)        
    col = np.arange(len(SGD_Confusion_Matrix.columns))
    true_sum = 0
    for c in col:
        print(c)
        true_sum = true_sum + SGD_Confusion_Matrix.iat[c,c]
    agreement_pct = true_sum /  SGD_Confusion_Matrix.sum().sum()
    print(agreement_pct)

# Test Lasso model
    clf = linear_model.Lasso(alpha = 0.1)
    clf.fit(training_set, training_labels)
    SGD_pred = clf.predict(test_set)
# Convert prediction back to GENHLTH score
    SGD_GENHLTH_pred = np.round(SGD_pred * GENHLTH_sd + GENHLTH_avg,0)
    GENHLTH_actual = np.round(test_labels * GENHLTH_sd + GENHLTH_avg,0)
# Confustion table
    SGD_Confusion_Matrix = DataFrame(confusion_matrix(SGD_GENHLTH_pred,GENHLTH_actual), index=[1,2,3,4,5], columns=[1,2,3,4,5])
    print(SGD_Confusion_Matrix)        
    col = np.arange(len(SGD_Confusion_Matrix.columns))
    true_sum = 0
    for c in col:
        print(c)
        true_sum = true_sum + SGD_Confusion_Matrix.iat[c,c]
    agreement_pct = true_sum /  SGD_Confusion_Matrix.sum().sum()
    print(agreement_pct)