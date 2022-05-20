# -*- coding: utf-8 -*-
"""
Created on Thu May 12 10:27:34 2022

@author: lgxsv2
"""
# SANDBOX File
# -*- coding: utf-8 -*-

#%% Packages

# general
import pandas as pd
import numpy as np
import time 
import matplotlib.pyplot as plt

#ML/AI/Stats packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import  Dense
from sklearn.model_selection import train_test_split


# For tensorboard: anaconda: cd to 00:code
# tensorboard --logdir=logs
from tensorflow.keras.callbacks import TensorBoard

#%% Data formatting

fn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\2022_03_martha\NN_dataset.csv"
ds = pd.read_csv(fn)
ds = ds.iloc[:,3:] # remove ID columns 
#predictor/labels split
y =  np.array(ds.iloc[:,9:]) 
X = np.array(ds.iloc[:,:9])
#Normalise(normalize)
X_normal = keras.utils.normalize(X)
y =  keras.utils.normalize(y)
X_train = X_normal
y_train = y
#%% Model Parameters

# final layer activation function - this is the output they want 
final_layer_AF = 'softmax'
#compiler
optim = 'adam'
loss = 'CategoricalCrossentropy'
metric = 'Accuracy'

epochs=10

#%% Model

# sequential feed forward ANN
model = keras.models.Sequential()

model.add(Dense(64, activation=('relu')))
# model.add(Dense(64, activation=('relu')))
# model.add(Dense(64, activation=('relu')))
#output
model.add(Dense(15, activation=(final_layer_AF)))

#compiler
model.compile(optimizer = optim, loss = loss, metrics = [metric])
#%%fit
history2 = model.fit(X_train, y_train, 
                    epochs=epochs,
                    batch_size=100,
                    validation_split=0.2)


#%%
# summarize history for accuracy
plt.figure()
plt.plot(history2.history['Accuracy'])
plt.plot(history2.history['val_Accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.figure()
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()































