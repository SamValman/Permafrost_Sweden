# -*- coding: utf-8 -*-
"""
###############################################################################
Project: SWEDISH PERMAFROST
###############################################################################
With ArcticDEM trainging data 
Created on Tue Jun 14 10:44:16 2022

@author: lgxsv2
"""


#%% Set up
import os
os.chdir(r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\Permafrost_Sweden')
get_ipython().run_line_magic('matplotlib', 'qt')

#%% Packages 
from PermafrostFunctionScript import view_argmax, view_band, predict_s2, beeper, saveTifs, join_rasters # implot, combine rasters
from tensorflow import keras
from tensorflow.keras.layers import  Dense

import time

# For tensorboard: anaconda: cd to 00:code
# tensorboard --logdir=logs
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np 
#%% Data
fn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\2022_06_DEM\ArcticDEMTrainingset\NN_dataset_DEMSlopeRough.csv"
ds = pd.read_csv(fn)

ds = ds.iloc[:,2:] # remove ID columns 

# #predictor/labels split
y =  np.array(ds.iloc[:,9:-3])

# The following 3 lines could probably be combined into a single line if I knew how 
X = np.array(ds.iloc[:,:9])
dem = np.array(ds.iloc[:,-3:])
                    
X = np.concatenate((X, dem), axis=1)

#%% max_membership

# amx = view_argmax(y)
# # import matplotlib.pyplot as plt
# # plt.figure()
# # unique, counts = np.unique(amx, return_counts=True)
# # plt.figure()
# # plt.bar(unique, counts)
# # plt.show()
# amx1 = amx.reshape(3560,1)
# X = np.concatenate((X, amx1),axis=1)

#%%


#Normalise(normalize)
X = keras.utils.normalize(X)
y =  keras.utils.normalize(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#%%

Name = "extraband{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir=f"logs/{Name}")

# one layer appears better 
#~280 ish epochs 
# softmax 1 is looking best, may change the kernel size

#compiler
optim = 'adam'
loss = 'CategoricalCrossentropy'
metric = 'Accuracy'

epochs=64

model = keras.models.Sequential()

model.add(Dense(8, activation=('relu')))


# model.add(Dense(128, activation=('softmax')))
# model.add(Dense(128, activation=('softmax')))

# model.add(Dense(128, activation=('softmax')))

#output
model.add(Dense(15, activation=('softmax')))

#compiler
model.compile(optimizer = optim, loss = loss, metrics = [metric])

#fit model
model.fit(X_train, y_train, epochs=epochs, batch_size=128, callbacks=[tensorboard])

# results
test_loss, test_accuracy = model.evaluate(X, y,callbacks=[tensorboard])
print ('test loss: ', test_loss)
print ('test accuracy:', test_accuracy)
#%%
# model.save('MR3')
#%%
internal_predict = model.predict(X_test, verbose=1, batch_size=128)

t = pd.DataFrame(internal_predict)
fn = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\Permafrost_Sweden\WorkingOutputs\2022_06_14_X.csv'
t.to_csv(fn)

t = pd.DataFrame(y_test)
fn = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\Permafrost_Sweden\WorkingOutputs\2022_06_14_Y.csv'
t.to_csv(fn)

