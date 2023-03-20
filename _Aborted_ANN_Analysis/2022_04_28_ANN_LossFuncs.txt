# -*- coding: utf-8 -*-
"""
###############################################################################
Project: SWEDISH PERMAFROST
###############################################################################

Created on Thu Apr 28 10:41:01 2022

@author: lgxsv2
"""
#%% Packages
print('''
      Two sections of script:
          - training and testing
          - Prediction 
          packages only needed for section 2 are loaded there. 
      ''')
# general
import pandas as pd
import numpy as np
import time 


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

X_train, X_test, y_train, y_test = train_test_split(X_normal, y, test_size=0.2, random_state=42)
#%%Here lies the issue
print('''
      The training dataset is heavily biased towards landcover 0
      we can see this if we argmax the set
      ''')
     
def argmax_a_prediction(prediction):
    '''
    name is a bit misleading
    will argmax any dataset/array

    '''
    ls = []
    for i in prediction:
        temp = np.argmax(i)
        ls.append(temp)
        #in case of an issue
        if type(temp)!=np.int64:
            print(temp)
            break
    return  np.array(ls)
y_argmax = argmax_a_prediction(y)
for i in range(15):
    print(y_argmax[y_argmax==i].shape)

#%% Model Parameters
'''
LIST OF ATTEMPTS (however stupid) in github repo under "2022_04_28_modelRuns.xlsx"
'''
# final layer activation function - this is the output they want 
final_layer_AF = 'softmax'
#compiler
optim = 'adam'
loss = 'CategoricalCrossentropy'
metric = 'Accuracy'

epochs=20

#%% Model
#TB organising
Name = "20mANN{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir=f"logs/{Name}")

# sequential feed forward ANN
model = keras.models.Sequential()

model.add(Dense(64, activation=('relu')))
model.add(Dense(64, activation=('relu')))
model.add(Dense(64, activation=('relu')))
#output
model.add(Dense(15, activation=(final_layer_AF)))

#compiler
model.compile(optimizer = optim, loss = loss, metrics = [metric])

#fit model
model.fit(X_train, y_train, epochs=epochs, batch_size=2, callbacks=[tensorboard])

# results
test_loss, test_accuracy = model.evaluate(X_test, y_test,callbacks=[tensorboard])
print ('test loss: ', test_loss)
print ('test accuracy:', test_accuracy)
#%%
# Predict the test set 
internal_predict = model.predict(X_test, verbose=1, batch_size=6)

#argmax result
internal_argmax = argmax_a_prediction(internal_predict)
print(np.unique(internal_argmax))
    

#%% SECTION 2: Predicting a new image (Whole S2 image)
# Takes ~1.5-2.5 hrs depending on cpu etc so don't run for now
print('''
      Predictions take 1.5-2.5 hours. Exiting script here.
      If you want to run, do so cell by cell below.
      ''')
raise SystemExit(0)

#%% packages 
import glob
import os
# Sat 
import skimage.io as IO
import matplotlib.pyplot as plt



def join_rasters(path):
    '''
    merges all jp2 images in a folder (path) and outputs a multibandArray
    path needs to be a real path r'//' 
    requires , subprocess, numpy as np, skimage.io as IO, and os
    '''
    ls =  glob.glob(os.path.join(path, '*.jp2'))
    im_ls = []
    for filename in ls:
        im = np.int16(IO.imread(filename))
        im = im.flatten()
        im_ls.append(im)
        
    multiBandImage = np.stack(im_ls)
    print(multiBandImage.shape)
    return multiBandImage

#%% sentinel 2 image
fn = "C:\\Users\\lgxsv2\\OneDrive - The University of Nottingham\\PhD\\yr_2\\01_RA2021_2022\\2022_03_arctic\\s2_files\\2019_07_27"
s2_2019_07_27 = join_rasters(fn)
#reshaping for model input
pX = np.reshape(s2_2019_07_27, (30140100, 9))
pX_normal = keras.utils.normalize(pX)

#%% run prediction

prediction = model.predict(pX_normal, verbose=1, batch_size=5)

# argmax result to see what its predicted
argmax_result = argmax_a_prediction(prediction)
print(np.unique(argmax_result))

#reshape back to s2 image
argmax_image = argmax_result.reshape((5490, 5490))

#again should be function but lets just visualise
plt.close('all')

def plotimage(im, title):
    '''
    plots sat image
    needs to be single band in desired shape
    '''
    # not sure if this is needed for the argmax/softmax bands
    im = keras.utils.normalize(im)
    
    plt.figure()
    plt.imshow(im) 
    plt.tile(title)
    plt.show() 
    
plotimage(argmax_image, 'argmax')
#%% Look at each band softmax value - saves individually 
def view_band(bname, im): 
    '''
    requires skimage.io as IO, os, np 
    prints band "bname" from multiband image "im"
    requires shape (30140100, bands) 
    '''
    band = im[:,bname].reshape(5490, 5490)
    plotimage(band, str(bname))
    

for i in range(15):
    view_band(i, prediction)
















