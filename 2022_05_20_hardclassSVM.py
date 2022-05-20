# -*- coding: utf-8 -*-
"""
###############################################################################
Project: SWEDISH PERMAFROST
###############################################################################
SVM 

Created on Fri May 20 11:58:07 2022

@author: lgxsv2
"""

import os
os.chdir(r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\Permafrost_Sweden')
get_ipython().run_line_magic('matplotlib', 'qt')
import pandas as pd 
import numpy as np 
import seaborn
import matplotlib.pyplot as plt
from PermafrostFunctionScript import view_argmax, view_band, predict_s2, beeper,predict_rf, saveTifs, join_rasters, implot # implot, combine rasters
from tensorflow import keras
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor

#%%
fn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\2022_SwedishPermafrostDataRepository\2022_05_16_updatedTrainingDataset.csv"
ds = pd.read_csv(fn)
ds = ds.iloc[:,2:] # remove ID columns 

#predictor/labels split
y =  np.array(ds.iloc[:,9:]) 
X = np.array(ds.iloc[:,:9])

X = keras.utils.normalize(X)
amx = view_argmax(y)


y = pd.get_dummies(amx)
y=np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#%%