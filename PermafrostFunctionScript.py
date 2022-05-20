# -*- coding: utf-8 -*-
"""
###############################################################################
Project: SWEDISH PERMAFROST
###############################################################################
Function script 

Created on Mon May  9 09:34:54 2022

@author: lgxsv2
"""
#%% Packages
# general
import numpy as np
import glob
import os
# Sat 
import skimage.io as IO
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
import datetime

#ML/AI/Stats packages
from tensorflow import keras
import winsound


def beeper(x):
    freq = 100
    dur = 500
  
    # loop iterates 5 times i.e, 5 beeps will be produced.
    for i in range(0, x):    
        winsound.Beep(freq, dur)    
        freq+= 100
        dur+= 50
        

def join_rasters(path):
    '''
    merges all jp2 images in a folder (path) and outputs a multibandArray
    path needs to be a real path r'//' 
    requires , subprocess, numpy as np, skimage.io as IO, and os
    '''
    im_ls = []
    x = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8a', 'B11', 'B12']
    for i in x:
        i = i+'.jp2'
        filename = os.path.join(path, i)
        print(filename)
        im = np.int16(IO.imread(filename))
        im = im.flatten()
        im_ls.append(im)
        
    multiBandImage = np.stack(im_ls)
    print(multiBandImage.shape)
    return multiBandImage

def implot(im, title='Satellite Image', normalize=False,sat=None,  prediction=True, save=False, fn=None):
    '''
    Plots image input 
    
    Parameters
    ----------
    im : Array
        Satellite image to plot, should work with RGB or prediction image.
    title : Str, 
        Will include a date on a line below regardless.
        The default is 'Satellite Image'.
    normalize : Boolean, optional
        images not in the 0-1 or 0-255 range require normlising.
        The default is False.
    sat: str
        reshape size 
        The default is None.
    prediction : Boolean, optional
        Is this a prediction e.g. 0-1 or an RGB. The default is True.
    save : Boolean, optional
        Save figure. The default is False.
    fn : Str, optional
        fn for saving image. The default is None.

    Returns
    -------
    fig : plt.fig
        figure returned for further alteration.
    
    '''
    if normalize:
        im = keras.utils.normalize(im)
        
    sat_dict = {'s2':(5490,5490), 'planet':(200,200)}
    
    if sat != None:
        im = im.reshape(sat_dict[sat])
        
    fig = plt.figure()
    
    title = title +'\n'+ str(datetime.datetime.now().date())
    plt.title(title)
    if prediction: 
        # cmap = colors.ListedColormap(['lightblue', 'green'])
        # label0 = mpatches.Patch(color='lightblue', label='River')
        # label1 = mpatches.Patch(color='green', label='Land')
        # plt.legend(handles=[label0, label1])

        plt.imshow(im) #  clim=(0,0.3)
    else:
        plt.imshow(im)
    
    plt.show()  
    
    if save:
        fn = fn+'.png'
        plt.savefig(fn, dpi=800)
        
    return fig



def view_band(bname, im): 
    '''
    requires skimage.io as IO, os, np 
    prints band "bname" from multiband image "im"
    requires shape (30140100, bands) 
    '''
    band = im[:,bname].reshape(5490, 5490)
    implot(band, str(bname), prediction=False)    

def view_argmax(prediction, plot=False):
    '''

    Parameters
    ----------
    prediction : array
        thin to be argmaxxed (assumes shape of S2 for print).
    plot: boolean
        default: False
        plots image 
    Returns
    -------
    just plots  atm .
    argmax result

    '''
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
    if plot:
        argmax_image = argmax_result.reshape((5490, 5490))
        implot(argmax_image, 'argmax image', prediction=False)
    return argmax_result
    
def predict_s2(fn, model):
    #load multiband raster (MBR)
    mbr = join_rasters(fn)

    
    # format
    pX = np.reshape(mbr, (30140100, 9))
    pX_normal = keras.utils.normalize(pX)
    
    # predict
    prediction = model.predict(pX_normal, verbose=1, batch_size=64)

    
    return prediction

def predict_rf(fn, model):
    #load multiband raster (MBR)
    mbr = join_rasters(fn)

    
    # format
    pX = np.reshape(mbr, (30140100, 9))
    pX_normal = keras.utils.normalize(pX)
    
    # predict
    prediction = model.predict(pX_normal)

    
    return prediction

def saveTifs(prediction, path):
    for i in range(15):
        band = prediction[:,i].reshape(5490, 5490)
        p = '2022_05_09_B'+str(i)+'.tif'
        bandpath = os.path.join(path,p )
        IO.imsave(bandpath, band) # imwrite may be better    

