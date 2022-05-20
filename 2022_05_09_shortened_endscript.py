# -*- coding: utf-8 -*-
"""
###############################################################################
Project: SWEDISH PERMAFROST
###############################################################################
Script to predict new S2 images

Created on Mon May  9 09:34:52 2022

@author: lgxsv2
"""
#%% Set up
import os
os.chdir(r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\Permafrost_Sweden')
get_ipython().run_line_magic('matplotlib', 'qt')

#%% Packages 
from PermafrostFunctionScript import view_argmax, view_band, predict_s2, beeper, saveTifs, join_rasters # implot, combine rasters
from tensorflow import keras

#%% load ANN model
# currently using MR1 which has a FCM output of 76.1%
model_fn = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\Permafrost_Sweden\MR1_76'
model = keras.models.load_model(model_fn)

#%% Sentinel files 
# S2B_MSIL2A_20190727T104029_N0213_R008_T34WDA_20190727T134833 = "C:\\Users\\lgxsv2\\OneDrive - The University of Nottingham\\PhD\\yr_2\\01_RA2021_2022\\2022_03_arctic\\s2_files\\2019_07_27"

#%% Carry out prediction
# prediction = predict_s2(S2B_MSIL2A_20190727T104029_N0213_R008_T34WDA_20190727T134833, model)

#%%
# beeper(10)
#%% look at each band
# view_argmax(prediction)    

# for i in range(15):
#     view_band(i, prediction)
    
#%% SAVE 
# create folders 
#'S2B_MSIL2A_20190727T104029_N0213_R008_T34WDA_20190727T134833',
# 'S2A_MSIL2A_20190726T102031_N0213_R065_T33WXR_20190726T125507',
# 'S2A_MSIL2A_20190726T102031_N0213_R065_T34WDA_20190726T125507',
# 'S2A_MSIL2A_20190726T102031_N0213_R065_T34WDB_20190726T125507',
# 'S2A_MSIL2A_20190726T102031_N0213_R065_T34WEA_20190726T125507',
# 'S2B_MSIL2A_20190727T104029_N0213_R008_T33WWR_20190727T134833',
# 'S2B_MSIL2A_20190727T104029_N0213_R008_T33WXR_20190727T134833',
# 'S2B_MSIL2A_20190727T104029_N0213_R008_T33WXS_20190727T134833',
# 'S2B_MSIL2A_20190727T104029_N0213_R008_T34WDB_20190727T134833',
# 'S2B_MSIL2A_20190727T104029_N0213_R008_T34WDV_20190727T134833',
# 'S2B_MSIL2A_20190727T104029_N0213_R008_T34WEA_20190727T134833',
# 'S2B_MSIL2A_20190727T104029_N0213_R008_T34WEB_20190727T134833',
# 'S2A_MSIL2A_20190728T105621_N0213_R094_T33WWQ_20190728T135007',
# 'S2A_MSIL2A_20190728T105621_N0213_R094_T33WWR_20190728T135007',


# 'S2B_MSIL2A_20190727T104029_N0213_R008_T33WXQ_20190727T134833',
# 'S2A_MSIL2A_20190729T103031_N0213_R108_T33WWR_20190729T133626',
listOfS2ProductNames= [
'S2B_MSIL2A_20190730T105039_N0213_R051_T33WWP_20190730T132548',
'S2B_MSIL2A_20190730T105039_N0213_R051_T33WWQ_20190730T132548',
'S2B_MSIL2A_20190730T105039_N0213_R051_T34WEA_20190730T132548']

#%%

for i in listOfS2ProductNames:
    granulefolder = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\s2_files'
    imf = os.path.join(granulefolder, i)
    # s2 = join_rasters(imf)
    
    prediction = predict_s2(imf, model)
    
    outfolder = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\2022_SwedishPermafrostDataRepository\computed_vegMAPS'
    of = os.path.join(outfolder, i)
    os.mkdir(of)

    saveTifs(prediction, of)
    
beeper(10)






