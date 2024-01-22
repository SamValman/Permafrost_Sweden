# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 04:25:33 2024

@author: lgxsv2
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

def sortData(fn, snow=False):
    # fn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\4&5Temperature, snowfallF2&3\\" + typ +"\\" + site +'.csv'
    df = pd.read_csv(fn)
    df['Datum'] = pd.to_datetime(df['Datum'], format='%d/%m/%Y')

    # Filter DataFrame for the range 2016 to 2021
    start_date = '2016-01-01'
    end_date = '2021-12-31'
    df = df[(df['Datum'] >= start_date) & (df['Datum'] <= end_date)]
    # df = df.resample('M', on='Datum').mean()
    print(df)

    x = df.iloc[:, 0]
    if snow:
        y = df.iloc[:, 1]
    else:
        y = df.iloc[:, 2]
        

    # do something
    return x, y



Kar = (r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\precipitation_data\Karesuando.csv")
#2016 to 2021
# sortData(Kar)


#%%

def shortTermWeather():
    fig, ax = plt.subplots(3,1)
    
    
    
    # air temperature##############################
    #file names
    a = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\4&5Temperature, snowfallF2&3\\temp\\abisko.csv"
    b = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\4&5Temperature, snowfallF2&3\\temp\\Karesuando.csv"
    c = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\4&5Temperature, snowfallF2&3\\temp\\Naimakka.csv"
    d = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\temperature_data\kiruna.csv"


    x,y = sortData(c)
    ax[0].plot(x, y, label='Naimakka',linestyle='--', color='black')
    x,y = sortData(b)
    ax[0].plot(x,y, label='Karesuando', linestyle=':', color='black')
    x,y = sortData(a)
    ax[0].plot(x,y, label='Abisko', linestyle='-', color='black')
    x,y = sortData(d)
    ax[0].plot(x,y, label='Kiruna', linestyle='-.', color='lightGrey')


    
    # snow ##########################
    a = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\4&5Temperature, snowfallF2&3\\snow\\abisko.csv"
    b = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\4&5Temperature, snowfallF2&3\\snow\\Karesuando.csv"
    c = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\4&5Temperature, snowfallF2&3\\snow\\Saarikoski.csv"
    d = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\snow_data\Kiruna.csv"

    # plot  
    x,y = sortData(c)
    ax[1].plot(x,y, label='Saarikoski', linestyle='--', color='black' )
    x,y = sortData(b)
    ax[1].plot(x,y, label='Karesuando', linestyle=':', color='black')
    x,y = sortData(a)
    ax[1].plot(x,y, label='Abisko', color='black')

    # precip
    # collect data 
    a = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\precipitation_data\Abisko.csv"
    b = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\precipitation_data\Naimakka.csv"
    c = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\precipitation_data\Karesuando_A.csv"
    d = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\precipitation_data\Kiruna.csv"

    # plot    
    x,y = sortData(b, True)
    ax[2].plot(x,y, label='Naimakka', linestyle='--', color='black' )
    x,y = sortData(c, True)
    ax[2].plot(x,y, label='Karesuando', linestyle=':', color='black')
    x,y = sortData(a, True)
    ax[2].plot(x,y, label='Abisko', color='black')
    x,y = sortData(d, True)
    ax[2].plot(x,y, label='Kiruna', linestyle='-', color='lightGrey')
    
    # axis titles 
    ax[0].set_ylabel('Air temperature (celsius)')
    ax[1].set_ylabel('Snow on ground (m)')
    ax[2].set_ylabel('Precipitation (mm)')

    ax[2].set_xlabel('Year recorded')


    # legends
    # ax[1].legend()

    
    plt.show()

shortTermWeather()