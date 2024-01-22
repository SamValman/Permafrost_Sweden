# -*- coding: utf-8 -*-
"""
##############################################################################
ARCTIC PERMAFROST: Temperature and Snow
##############################################################################
All bar needed for paper commented out 
Created on Mon Jul 18 13:57:35 2022

@author: lgxsv2

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
get_ipython().run_line_magic('matplotlib', 'qt')

#%%





def files(df=None, typ='snow', site='abisko'):
    try: len(df)
    except TypeError:
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
def panel(tn1,tk1,tk2,ta1,tkir,  ss1,sk1,sk2,sa1, skir, Na, Kar, Kat, Ab, kir):
    '''
    temp, snow, precip
    '''    
    fig, ax = plt.subplots(1,3, figsize=(15,6))
    
    ax[0].plot(tn1['Datum'], tn1['max'], label='Naimakka',linestyle='--', color='black')
    ax[0].plot(tk1['Datum'], tk1['max'], label='Karesuando', linestyle=':', color='black')
    # ax[0].plot(tk2['Datum'], tk2['max'], label='Katterjåkk', linestyle='-.', color='black')
    ax[0].plot(ta1['Datum'], ta1['max'], label='Abisko', linestyle='-', color='black')
    ax[0].plot(tkir['Datum'], tkir['max'], label='Kiruna', linestyle='-', color='lightGrey')
    ax[0].legend(loc='best', bbox_to_anchor=(0.45,0.25))
    ax[0].set_ylabel('Air temperature (celsius)')
    ax[0].set_xlabel('Year recorded')
    plt.text(-9400, 3.35, 'a)')
    plt.text(850, 3.35, 'b)')
    plt.text(11000,3.35, 'c)')


    ax[1].plot(ss1['Datum'], ss1['Snödjup'], label='Saarikoski', linestyle='--', color='black' )
    ax[1].plot(sk1['Datum'], sk1['Snödjup'], label='Karesuando', linestyle=':', color='black')
    # ax[1].plot(sk2['Datum'], sk2['Snödjup'], label='Katterjåkk', linestyle='-.', color='black')
    ax[1].plot(sa1['Datum'], sa1['Snödjup'], label='Abisko', color='black')
    ax[1].legend()
    ax[1].set_ylabel('Snow on ground (m)')
    ax[1].set_xlabel('Year recorded')
    
    
    
    
    ax[2].plot(Na['Datum'], Na['precip_mm'], label='Naimakka', linestyle='--', color='black' )
    ax[2].plot(Kar['Datum'], Kar['precip_mm'], label='Karesuando', linestyle=':', color='black')
    # ax[2].plot(Kat['Datum'], Kat['precip_mm'], label='Katterjåkk', linestyle='-.', color='black')
    ax[2].plot(Ab['Datum'], Ab['precip_mm'], label='Abisko', color='black')
    ax[2].plot(kir['Datum'], kir['precip_mm'], label='Kiruna', linestyle='-', color='lightGrey')

    # ax[1].plot(skir['Datum'], skir['Snödjup'], label='Kiruna', linestyle='-', color='lightGrey')
    ax[2].legend()
    ax[2].set_ylabel('Precipitation (mm)')
    ax[2].set_xlabel('Year recorded')

    plt.show()
    

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


# remove erronous values where part year 
temp_k1 = temp_k1.iloc[1: , :]
temp_k2 = temp_k2.iloc[:-1 , :]

#%% sorting kir




kir_t = pd.read_csv(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\temperature_data\kiruna.csv")
kir_t = files(kir_t, typ='temp')


kir_s = pd.read_csv(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\snow_data\Kiruna.csv")
kir_s = files(kir_s, typ='snow')

#%%

#Precipitation
Kar = pd.read_csv(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\precipitation_data\Karesuando.csv")
Kar = files(Kar, typ='pcip') # missing 13, 14,15,16,17, 18
Kar = Kar.sort_values(by='Datum')
Kar = Kar[Kar['Datum']<'2012-12-31']

# [2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2019 2020 2021]
KarA = pd.read_csv(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\precipitation_data\Karesuando_A.csv")
KarA = files(KarA, typ='pcip') # to fill in gaps in above record
KarA = KarA.sort_values(by='Datum')
KarA = KarA[KarA['Datum']>'2012-12-31']

Kar = Kar.append(KarA)

Kat = pd.read_csv(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\precipitation_data\Katterjakk.csv")
Kat = files(Kat, typ='pcip')

Na = pd.read_csv(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\precipitation_data\Naimakka.csv")
Na = files(Na, typ='pcip')
Ab = pd.read_csv(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\precipitation_data\Abisko.csv")
Ab = files(Ab, typ='pcip') 

##### - then add to image and see how it looks
kir = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\2023_04_25_ClimateDataFile\f4\precipitation_data\Kiruna.csv"
kir = pd.read_csv(kir)
kir = files(kir, typ='pcip')
#%%
plt.close()
panel(temp_n1, temp_k1, temp_k2, temp_a1, kir_t, snow_s1, snow_k1, snow_k2, snow_a1, kir_s, Na, Kar, Kat, Ab, kir)
fn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\f2\2024_01_18_F7.jpeg"
plt.savefig(fn,dpi=800 )
# add kiruna 












































#%%

# plt.figure()
# plt.plot(n1['Datum'], n1['max'], label='Naimakka',linestyle='--', color='black')
# plt.plot(k1['Datum'], k1['max'], label='Karesuando', linestyle=':', color='black')
# plt.plot(k2['Datum'], k2['max'], label='Katterjåkk', linestyle='-.', color='black')
# plt.plot(a1['Datum'], a1['max'], label='Abisko', linestyle='-', color='black')
# plt.legend()

# plt.ylabel('Air temperature (celsius)')
# plt.xlabel('Year recorded')
























# plt.figure()
# # 

# plt.plot(s1['Datum'], s1['Snödjup'], label='Saarikoski', linestyle='--', color='black' )
# plt.plot(k1['Datum'], k1['Snödjup'], label='Karesuando', linestyle=':', color='black')
# plt.plot(k2['Datum'], k2['Snödjup'], label='Katterjåkk', linestyle='-.', color='black')
# plt.plot(a1['Datum'], a1['Snödjup'], label='Abisko', color='black')
# plt.legend()

# plt.ylabel('Snow on ground (cm)')
# plt.xlabel('Year recorded')
# plt.title('Average annual snow on ground')
# plt.show()
# output = r'C:\Users\lgxsv2\Downloads\snow\2020_07_25\w.png'
# output = r'C:\Users\lgxsv2\Downloads\snow\2022_07_25\snow_annualOneGraph_no0_dash.png'
# # plt.savefig(output, dpi=600)


# #######################################################################
# #%% TEMPERATURE
# a1 = files(typ='temp', site='abisko')
# k1 = files(typ='temp', site='Karesuando')
# k2 = files(typ='temp', site='Katterjakk')
# n1 = files(typ='temp', site='Naimakka')

# k1 = k1.iloc[1: , :]
# k2 = k2.iloc[:-1 , :]


# #%%
# plt.figure()
# plt.plot(n1['Datum'], n1['max'], label='Naimakka',linestyle='--', color='black')
# plt.plot(k1['Datum'], k1['max'], label='Karesuando', linestyle=':', color='black')
# plt.plot(k2['Datum'], k2['max'], label='Katterjåkk', linestyle='-.', color='black')
# plt.plot(a1['Datum'], a1['max'], label='Abisko', linestyle='-', color='black')
# plt.legend()

# plt.ylabel('Air temperature (celsius)')
# plt.xlabel('Year recorded')
# plt.title('Mean annual air temperature')
# plt.show()
# output = r'C:\Users\lgxsv2\Downloads\temp\2022_07_25\tempyr_max_singlegraph_black.png'

# raise SystemExit()





#%%


    

# def fig(s1,k1,k2,a1, title, x_lab, output='', save='no'):
#     plt.close('all')
    
#     s1 = s1.reset_index()
#     k1 = k1.reset_index()
#     k2 = k2.reset_index()
#     a1 = a1.reset_index()
    
    
    
    
    
    
    
#     fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    
#     ax[0,0].plot(s1['Datum'], s1['Snödjup'])
#     ax[0,1].plot(k1['Datum'], k1['Snödjup'])
#     ax[1,0].plot(k2['Datum'], k2['Snödjup'])
#     ax[1,1].plot(a1['Datum'], a1['Snödjup'])
    
#     ax[0,0].set_title('Saarikoski')
#     ax[0,1].set_title('Karesuando')
#     ax[1,0].set_title('Katterjarkk')
#     ax[1,1].set_title('Abisko')
    
#     fig.suptitle(title, weight='bold')
#     plt.text(-100, 1.2,x_lab,  rotation='vertical', weight='bold')
#     if save=='yes':
#         plt.savefig(output, dpi=600)
#     plt.show()




#%%



# plt.figure()
# plt.scatter(s1['Datum'], s1['Snödjup'])



#%%

# output = r'C:\Users\lgxsv2\Downloads\snow\2022_07_25\snow_yrAvg.png'

# fig(s1,k1,k2,a1,'Snow on Ground', 'Snow depth (cm)', output, 'yes' )

#%%


# #%%

# output = r'C:\Users\lgxsv2\Downloads\snow\2022_07_25\snow_annualMax_no.png'

# fig(s1,k1,k2,a1,'Annual max snow on ground', 'Snow depth (cm)', output, 'yes' )

# #%%
# output = r'C:\Users\lgxsv2\Downloads\snow\2022_07_25_imgs\snow_annualMaxline_no0.png'

# fig(s1,k1,k2,a1,'Annual max snow on ground', 'Snow depth (cm)', output, 'yes' )
























# #%%
# # temperature

# def importtemp(fn):
#     df = pd.read_csv(fn)
#     df = df.dropna()
#     df['Datum'] =  pd.to_datetime(df['Datum'], format='%d/%m/%Y')
#     df = df[(df['Datum'] > '2000-01-01') & (df['Datum'] < '2022-01-01')]
#     return df 

# fn = r"C:\Users\lgxsv2\Downloads\temp\Abisko.csv"

# a1 = importtemp(fn)
# fn = r"C:\Users\lgxsv2\Downloads\temp\Karesuando.csv"
# k1 = importtemp(fn)
# fn = r"C:\Users\lgxsv2\Downloads\temp\Katterjakk.csv"
# k2 = importtemp(fn)
# fn = r"C:\Users\lgxsv2\Downloads\temp\pajala.csv"
# p1 = importtemp(fn)

# #%%
# # def temperatureFigure(p1,k1,k2,a1, title, x_lab,m='max', output='', save='no'):
# #     plt.close('all')
    
# #     fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    
# #     ax[0,0].scatter(p1['Datum'], p1[m])
# #     ax[0,1].scatter(k1['Datum'], k1[m])
# #     ax[1,0].scatter(k2['Datum'], k2[m])
# #     ax[1,1].scatter(a1['Datum'], a1[m])
    
# #     ax[0,0].set_title('Pajala')
# #     ax[0,1].set_title('Karesuando')
# #     ax[1,0].set_title('Katterjarkk')
# #     ax[1,1].set_title('Abisko')
    
# #     fig.suptitle(title, weight='bold')
# #     plt.text(-100, 1.2,x_lab,  rotation='vertical', weight='bold')
# #     if save=='yes':
# #         plt.savefig(output, dpi=600)
# #     plt.show()
    
# # output = r'C:\Users\lgxsv2\Downloads\temp\2022_07_20_imgs\tempAll_max.png'
# # temperatureFigure(p1,k1,k2,a1, 'Temperature max', 'celcius', m='max', output=output, save='yes')
# # output = r'C:\Users\lgxsv2\Downloads\temp\2022_07_20_imgs\tempAll_min.png'
# # temperatureFigure(p1,k1,k2,a1, 'Temperature min', 'celcius', m='min', output=output, save='yes')

# #%% files(typ='snow', site='abisko')

# p1 = p1.resample('Y', on='Datum').mean()
# k1 = k1.resample('Y', on='Datum').mean()
# k2 = k2.resample('Y', on='Datum').mean()
# a1 = a1.resample('Y', on='Datum').mean()

# # output = r'C:\Users\lgxsv2\Downloads\temp\2022_07_20_imgs\tempyr_max.png'
# # temperatureFigure(p1,k1,k2,a1, 'Temperature max yearly', 'celcius', m='max', output=output, save='yes')
# # output = r'C:\Users\lgxsv2\Downloads\temp\2022_07_20_imgs\tempyr_min.png'
# # temperatureFigure(p1,k1,k2,a1, 'Temperature min yearly', 'celcius', m='min', output=output, save='yes')
# #%%
# p1 = p1.reset_index()
# k1 = k1.reset_index()
# k2 = k2.reset_index()
# a1 = a1.reset_index()

# # last value removed as incomplete measurements
# k2 = k2[:-1] 

# #%%
# # def temperatureFigure(p1,k1,k2,a1, title, x_lab,m='max', output='', save='no'):
# #     plt.close('all')
    
# #     fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    
# #     ax[0,0].plot(p1['Datum'], p1[m])
# #     ax[0,1].plot(k1['Datum'], k1[m])
# #     ax[1,0].plot(k2['Datum'], k2[m])
# #     ax[1,1].plot(a1['Datum'], a1[m])
    
# #     ax[0,0].set_title('Pajala')
# #     ax[0,1].set_title('Karesuando')
# #     ax[1,0].set_title('Katterjarkk')
# #     ax[1,1].set_title('Abisko')
    
# #     fig.suptitle(title, weight='bold')
# #     plt.text(-100, 1.2,x_lab,  rotation='vertical', weight='bold')
# #     if save=='yes':
# #         plt.savefig(output, dpi=600)
# #     plt.show()

# # output = r'C:\Users\lgxsv2\Downloads\temp\2022_07_20_imgs\tempyr_max_line.png'
# # temperatureFigure(p1,k1,k2,a1, 'Temperature max yearly', 'celcius', m='max', output=output, save='yes')
# # output = r'C:\Users\lgxsv2\Downloads\temp\2022_07_20_imgs\tempyr_min_line.png'
# # temperatureFigure(p1,k1,k2,a1, 'Temperature min yearly', 'celcius', m='min', output=output, save='yes')

# #%%
# # plt.figure()
# # plt.plot(p1['Datum'], p1['min'], label='Pajala')
# # plt.plot(k1['Datum'], k1['min'], label='Karesuando')
# # plt.plot(k2['Datum'], k2['min'], label='Katterjarkk')
# # plt.plot(a1['Datum'], a1['min'], label='Abisko')
# # plt.legend()

# # plt.xlabel('Air temperature (celsius)')
# # plt.title('Min air temperature')
# # plt.show()
# # output = r'C:\Users\lgxsv2\Downloads\temp\2022_07_20_imgs\tempyr_min_singlegraph.png'

# # plt.savefig(output, dpi=600)

# #%%
# plt.figure()
# plt.plot(p1['Datum'], p1['max'], label='Pajala',linestyle='--', color='black')
# plt.plot(k1['Datum'], k1['max'], label='Karesuando', linestyle=':', color='black')
# plt.plot(k2['Datum'], k2['max'], label='Katterjarkk', linestyle='-.', color='black')
# plt.plot(a1['Datum'], a1['max'], label='Abisko', linestyle='-', color='black')
# plt.legend()

# plt.ylabel('Air temperature (celsius)')
# plt.xlabel('Year recorded')
# plt.title('Mean annual air temperature')
# plt.show()
# output = r'C:\Users\lgxsv2\Downloads\temp\2022_07_25\tempyr_max_singlegraph_black.png'

# # plt.savefig(output, dpi=600)
