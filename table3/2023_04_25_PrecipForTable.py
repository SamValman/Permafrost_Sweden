# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:35:04 2023

@author: lgxsv2
"""

import pandas as pd
import numpy as np 

def files(df):



    df = df.dropna()
    df['Datum'] =  pd.to_datetime(df['Datum'], format='%d/%m/%Y')
    df = df[(df['Datum'] > '2000-01-01') & (df['Datum'] < '2022-01-01')]


    df = df.reset_index(drop=True)

    return df

#%% get the six stations using the files method above 

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

#%%
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
    
# printStats(Ab1)

#%%
printStats(Ab)

#%%
















































































































































#Not entirely sure where Paj came from 
# Pa = pd.read_csv(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\precipitation_data\Pajala.csv")
# Pa = files(Pa) # only till 2008 # remove 08
# PaA = pd.read_csv(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\precipitation_data\Pajala_A.csv")
# PaA = files(PaA) # fill in from 2008


# Ab1 = printStats(Kar)


# Kiruna snowfall and temp
kir_t = pd.read_csv(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\temperature_data\kiruna.csv")
kir_t = files(kir_t)
printStats(kir_t)


#%% SNOW
def files(df, typ='snow'):
    



    df = df.dropna()
    df['Datum'] =  pd.to_datetime(df['Datum'], format='%d/%m/%Y')
    df = df[(df['Datum'] > '2000-01-01') & (df['Datum'] < '2022-01-01')]

    df.loc[(df!=0).any(axis=1)]
    # if typ=='snow':
    #     df = df.resample('Y', on='Datum').sum()
    
    #     df['Snödjup'] = df['Snödjup']/365
    

    # # df = df.loc[(df!=0).any(axis=1)]
    #     df.replace(0, np.nan, inplace=True)
    # else:
    #     df = df.resample('Y', on='Datum').mean()

    # df = df.reset_index()

    return df

kir_s = pd.read_csv(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\snow_data\Kiruna.csv")
kir_s = files(kir_s)
printStats(kir_s)