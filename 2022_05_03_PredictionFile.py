# -*- coding: utf-8 -*-
"""
###############################################################################
Project: SWEDISH PERMAFROST
###############################################################################
Script to predict new S2 images

Created on Thu Apr 28 10:41:01 2022

@author: lgxsv2
"""
#%% Packages
# general
import pandas as pd
import numpy as np
import time 
import glob
import os
# Sat 
import skimage.io as IO
import matplotlib.pyplot as plt

#ML/AI/Stats packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import  Dense
from sklearn.model_selection import train_test_split


# For tensorboard: anaconda: cd to 00:code
# tensorboard --logdir=logs
from tensorflow.keras.callbacks import TensorBoard
#%% load ANN model
# currently using MR1 which has a FCM output of 76.1%
model_fn = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\Permafrost_Sweden\MR1_76'
model = keras.models.load_model(model_fn)
#%% Helper functions

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

def view_band(bname, im): 
    '''
    requires skimage.io as IO, os, np 
    prints band "bname" from multiband image "im"
    requires shape (30140100, bands) 
    '''
    band = im[:,bname].reshape(5490, 5490)
    plotimage(band, str(bname))    

def view_argmax(prediction):
    # argmax prediction
    ls = []
    for i in prediction:
        temp = np.argmax(i)
        ls.append(temp)
        #in case of an issue
        if type(temp)!=np.int64:
            print(temp)
            break
    argmax_result = np.array(ls)
    argmax_image = argmax_result.reshape((5490, 5490))
    plotimage(argmax_image, 'argmax image')

#%% Sentinel files 
s07_27 = "C:\\Users\\lgxsv2\\OneDrive - The University of Nottingham\\PhD\\yr_2\\01_RA2021_2022\\2022_03_arctic\\s2_files\\2019_07_27"

    
#%% Prediction function

def predict_s2(fn, model):
    #load multiband raster (MBR)
    mbr = join_rasters(fn)

    
    # format
    pX = np.reshape(mbr, (30140100, 9))
    pX_normal = keras.utils.normalize(pX)
    
    # predict
    prediction = model.predict(pX_normal, verbose=1, batch_size=64)

    
    return prediction

#%% Carry out prediction
prediction = predict_s2(s07_27, model)

#%% look at each band
view_argmax(prediction)    

for i in range(15):
    view_band(i, prediction)








