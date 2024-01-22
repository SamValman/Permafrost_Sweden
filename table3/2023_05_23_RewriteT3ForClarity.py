# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:38:05 2023

@author: lgxsv2
"""

# Section ONE will be precipitation. TWO snow, THREE Temperature
# the first part will run
# the second part should be run itterative to print results

#%% packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import os
get_ipython().run_line_magic('matplotlib', 'qt')

#%% Section ONE Precipitation: functions

def files(df):



    df = df.dropna()
    df['Datum'] =  pd.to_datetime(df['Datum'], format='%d/%m/%Y')
    df = df[(df['Datum'] > '2000-01-01') & (df['Datum'] < '2022-01-01')]


    df = df.reset_index(drop=True)

    return df

def printStats(df,start_slice=0, end_slice=None ):
    
    r = df.groupby(df.Datum.dt.to_period("Y"))['precip_mm'].sum()
    years = df.groupby(df.Datum.dt.to_period("Y"))['precip_mm'].count()
    
    print(years)
    
    r = r[start_slice:end_slice]
    var = df.iloc[:,-1]
    
    print('mean: ', r.mean())
    print('median: ', var.median())
    mi = var.min()
    ma = var.max()
    print('min: ', mi)
    print('max: ', ma)
    lq, uq = np.percentile(var, [25,75])
    print('range: ', uq-lq)
    

#%% Section ONE: load in sites
'''
Precip:
Kir only to 2009. 
Sa 2007 to 2021

'''

Ab = pd.read_csv(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\precipitation_data\Abisko.csv")
Ab = files(Ab) 


Kar = pd.read_csv(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\precipitation_data\Karesuando.csv")
Kar = files(Kar) # missing 13, 14,15,16,17, 18
Kar = Kar.sort_values(by='Datum')
Kar = Kar[Kar['Datum']<'2012-12-31']

# [2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2019 2020 2021]
KarA = pd.read_csv(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\precipitation_data\Karesuando_A.csv")
KarA = files(KarA) # to fill in gaps in above record
KarA = KarA.sort_values(by='Datum')
KarA = KarA[KarA['Datum']>'2012-12-31']

Kar = Kar.append(KarA)

#[2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021]


Kat = pd.read_csv(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\precipitation_data\Katterjakk.csv")
Kat = files(Kat)

Na = pd.read_csv(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\precipitation_data\Naimakka.csv")
Na = files(Na)

Kir = pd.read_csv(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\precipitation_data\Kiruna.csv")
Kir = files(Kir)



Sa = pd.read_csv(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\precipitation_data\Saarikoski.csv")
Sa = files(Sa) # missing 00-06 inclusive # not solvable 

#%% Section ONE: b: run itteratively for precip
printStats(Sa)












#####################################################################################################################
#%% Section TWO: SNOW
def files(site='abisko'):
    
    fn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\snow_data"
    df = pd.read_csv(os.path.join(fn, (site+'.csv')))


    # df = df.dropna()
    df['Datum'] =  pd.to_datetime(df['Datum'], format='%d/%m/%Y')
    df = df[(df['Datum'] > '2000-01-01') & (df['Datum'] < '2022-01-01')]

    # df = df.resample('Y', on='Datum').sum()
    
    df = df.reset_index()

    return df



def printStats(df):
    
    years = df.groupby(df.Datum.dt.to_period("Y"))['Snödjup'].count()
    print(years)
    
    tot = df.groupby(df.Datum.dt.to_period("Y"))['Snödjup'].sum()
    
    var = df.iloc[:,-1]
    print('mean: ', tot.mean())

    ma = var.max()
    print('max: ', ma)
    var = var.dropna()
    lq, uq = np.percentile(var, [25,75])
    
    print('range: ', uq-lq)


#%%
# Na_s = files(site='') # no snow data
Sa_s = files(site='Saarikoski')
Kar_s = files(site='Karesuando')
Kat_s = files(site='Katterjarkk')
Ab_s = files()
Kir_s = files(site='Kiruna')
Kir_s = Kir_s[Kir_s['Datum'].dt.year != 2010]


#%%  
# printStats(Sa_s)
# printStats(Kar_s)
# printStats(Kat_s)
# printStats(Ab_s)
printStats(Kir_s) # 5

#####################################################################################################################
#%% Section THREE temperature 
def files(site='abisko'):
    
    fn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\temperature_data"
    df = pd.read_csv(os.path.join(fn, (site+'.csv')))


    # df = df.dropna()
    df['Datum'] =  pd.to_datetime(df['Datum'], format='%d/%m/%Y')
    df = df[(df['Datum'] > '2000-01-01') & (df['Datum'] < '2022-01-01')]

    # df = df.resample('Y', on='Datum').sum()
    
    df = df.reset_index()

    return df



def printStats(df):
    
    years = df.groupby(df.Datum.dt.to_period("Y"))['max'].count()
    print(years)
    
    tot = df.groupby(df.Datum.dt.to_period("Y"))['max'].mean()
    
    var = df['max']
    print('mean: ', tot.mean())

    ma = var.max()
    print('max: ', ma)
    print('min', var.min())
    var = var.dropna()
    lq, uq = np.percentile(var, [25,75])
    
    print('range: ', uq-lq)


#%%
Na_t = files( site='Naimakka')
Kar_t = files( site='Karesuando')
Kar_t = Kar_t[Kar_t['Datum'].dt.year != 2008]

Kat_t = files( site='Katterjakk')
Kat_t = Kat_t[Kat_t['Datum'].dt.year != 2019]

Ab_t = files(site='Abisko')
Kir_t = files(site='Kiruna')

#%%
# printStats(Na_t)
# printStats(Kar_t)
# printStats(Kat_t)
# printStats(Ab_t)
printStats(Kir_t) # 5


#%% HARD CODING temperature averages


def files(df):



    df = df.dropna()
    df['Datum'] =  pd.to_datetime(df['Datum'], format='%d/%m/%Y')
    df = df[(df['Datum'] > '2000-01-01') & (df['Datum'] < '2022-01-01')]


    df = df.reset_index(drop=True)

    return df


Na = files(pd.read_csv(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\temperature_data\AVG_Naimakka.csv"))
Kar = files(pd.read_csv(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\temperature_data\AVG_Karesuando.csv"))
Kat = files(pd.read_csv(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\temperature_data\AVG_Katterjakk.csv"))
Ab = files(pd.read_csv(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\temperature_data\AVG_Abisko.csv"))
Kir = files(pd.read_csv(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\temperature_data\AVG_Kiruna.csv"))


def printStats(df):
    
    years = df.groupby(df.Datum.dt.to_period("Y"))['t'].count()
    print(years)
    
    tot = df.groupby(df.Datum.dt.to_period("Y"))['t'].mean()
    
    print('mean: ', tot.mean())

printStats(Ab)


