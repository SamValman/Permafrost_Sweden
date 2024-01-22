# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 09:07:38 2022

@author: lgxsv2
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
get_ipython().run_line_magic('matplotlib', 'qt')

#%%





def files(typ='snow', site='abisko'):
    
    fn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\4&5Temperature, snowfallF2&3\\" + typ +"\\" + site +'.csv'
    df = pd.read_csv(fn)


    df = df.dropna()
    df['Datum'] =  pd.to_datetime(df['Datum'], format='%d/%m/%Y')
    df = df[(df['Datum'] > '2000-01-01') & (df['Datum'] < '2022-01-01')]

    df.loc[(df!=0).any(axis=1)]
    if typ=='snow':
        df = df.resample('Y', on='Datum').sum()
    
        df['Snödjup'] = df['Snödjup']/365
    

    # df = df.loc[(df!=0).any(axis=1)]
        df.replace(0, np.nan, inplace=True)
    else:
        df = df.resample('Y', on='Datum').mean()

    df = df.reset_index()

    return df

#%% Snow data
snow_a1 = files(typ='snow', site='abisko')
snow_k1 = files(typ='snow', site='Karesuando')
snow_k2 = files(typ='snow', site='Katterjarkk')
snow_s1 = files(typ='snow', site='Saarikoski')

##
temp_a1 = files(typ='temp', site='abisko')
temp_k1 = files(typ='temp', site='Karesuando')
temp_k2 = files(typ='temp', site='Katterjakk')
temp_n1 = files(typ='temp', site='Naimakka')
# temp_kiruna = files(typ='temp', site='kiruna')


# remove erronous values where part year 
temp_k1 = temp_k1.iloc[1: , :]
temp_k2 = temp_k2.iloc[:-1 , :]


#%%


def printStats(df):
    
    var = df.iloc[:,-1]
    print('mean: ', var.mean())
    print('median: ', var.median())
    mi = var.min()
    ma = var.max()
    print('min: ', mi)
    print('max: ', ma)
    lq, uq = np.percentile(var, [25,75])
    print('range: ', uq-lq)
    
printStats(snow_a1)
















