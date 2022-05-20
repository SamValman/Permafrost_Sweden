# -*- coding: utf-8 -*-
"""
###############################################################################
Project: SWEDISH PERMAFROST
###############################################################################
hard classification using random forest
Created on Thu May 19 16:06:15 2022

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


fn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\2022_SwedishPermafrostDataRepository\2022_05_16_updatedTrainingDataset.csv"
ds = pd.read_csv(fn)
ds = ds.iloc[:,2:] # remove ID columns 

#predictor/labels split
y =  np.array(ds.iloc[:,9:]) 
X = np.array(ds.iloc[:,:9])

#%% max_membership
X = keras.utils.normalize(X)
amx = view_argmax(y)
#%% TTS
y = pd.get_dummies(amx)
y=np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Random forest 

# Import the model we are using
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1500, random_state = 42)
# Train the model on training data
z = rf.fit(X_train, y_train)


#%%

predictions = z.predict(X_test)
internal_predict = predictions
predictions = view_argmax(predictions)
y_test = view_argmax(y_test)
conf_mat = confusion_matrix(y_test, predictions)
print(conf_mat)

seaborn.heatmap(conf_mat)
plt.show()

#%%
# Accuracy
print('ac', accuracy_score(y_test, predictions))

# Recall
# from sklearn.metrics import recall_score
# print(recall_score(y_test, predictions, average='weighted'))
# # Precision
# from sklearn.metrics import precision_score
# print(precision_score(y_test, predictions, average='weighted'))

# from sklearn.metrics import specificity_score
#%%
i='S2B_MSIL2A_20190727T104029_N0213_R008_T34WDA_20190727T134833'
granulefolder = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\s2_files'
imf = os.path.join(granulefolder, i)

prediction = predict_rf(imf, rf)
#%%
p = view_argmax(prediction)

#%%
# saveTifs(prediction, of)

outputit = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\Permafrost_Sweden\WorkingOutputs\2022_05_20_RF_im'

implot(p, title='Random forest classification', sat='s2', save=True, fn=outputit )
#%%







#%% sizes 
files = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\Permafrost_Sweden\WorkingOutputs\bands\2022_04_25_B0.tif"
import skimage.io as IO
im = np.int16(IO.imread(files))

#%% 
outputit = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\Permafrost_Sweden\WorkingOutputs\2022_05_20_RF.tif'
IO.imsave( outputit, p.reshape(5490,5490))

#%%

plt.figure()
labels, counts = np.unique(amx, return_counts=True)

plt.bar(labels, counts)
plt.ylabel('class')
plt.xlabel('count')
plt.title('training data argmaxed')
plt.show()




#%%
a_train, a_test, b_train, b_test = train_test_split(X, y, test_size=0.2, random_state=42)



t = pd.DataFrame(internal_predict)
fn = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\Permafrost_Sweden\WorkingOutputs\rf_p.csv'
t.to_csv(fn)

t = pd.DataFrame(b_test)
fn = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\Permafrost_Sweden\WorkingOutputs\rf_Y.csv'
t.to_csv(fn)