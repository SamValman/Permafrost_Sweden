# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:22:17 2022

@author: lgxsv2
"""

import numpy as np 
import skimage.io as IO
import pandas as pd
import matplotlib.pyplot as plt

#%%
fn = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\00_code\outputs\2022_04_21_ls.npy'
ls = np.load(fn)
fn = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\00_code\outputs\2022_04_21_predicted_image.npy'
predicted_image = np.load(fn)

fn = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\00_code\outputs\2022_04_21_prediction.npy'
prediction = np.load(fn)

#%%
# These has been argmaxed (15 bands only max kept)
print(type(ls), ls.shape)
print(type(predicted_image), predicted_image.shape) 
# This one has not 
print(type(prediction), prediction.shape)

#%%
# shape wanted: (5490, 5490, 15)
multiBandIm = prediction.reshape(5490,5490,15)

#%%
# Try and save this image as a multiband image:
#     attempt 1: imsave basic 
fname = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\Permafrost_Sweden\WorkingOutputs\2022_04_25_multiBandIm.tif'
IO.imsave(fname, multiBandIm)

# Result: 15 accross single band
#   attempt 2: try to reshape

multiBandIm = prediction.reshape(15, 5490,5490)
IO.imsave(fname, multiBandIm)
# Result: 1 band lined image mult
#%%
    # attempt 3: use tifffile :https://stackoverflow.com/questions/53776506/how-to-save-an-array-representing-an-image-with-40-band-to-a-tif-file
# was already downloaded (must have been a requirement for another package)
import tifffile
# example suggests requirement for bands first
fname = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\Permafrost_Sweden\WorkingOutputs\2022_04_25_trifffileMBI.tif'

tifffile.imwrite(fname, multiBandIm)
# same problem


#%%

    # attempt 4: save individual then use gdal to combine. 
# multiBandIm = prediction.reshape(5490,5490,15)
# b1 = prediction[:, 0].reshape(5490, 5490)
    


# fname = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\Permafrost_Sweden\WorkingOutputs\2022_04_25_B1_test.tif'
# IO.imsave(fname, b1)
def pick_band(bname): 
    '''
    requires skimage.io as IO, os, np 
    '''
    import os
    
    
    band = prediction[:,bname].reshape(5490, 5490)
    path = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\Permafrost_Sweden\WorkingOutputs\bands'
    p = '2022_04_25_B'+str(bname)+'.tif'
    path = os.path.join(path,p )
    IO.imsave(path, band)

for i in range(15):
    pick_band(i)

#%%
# Now combine them
from osgeo import gdal



import glob
import os

path = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\Permafrost_Sweden\WorkingOutputs\bands'
output = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\Permafrost_Sweden\WorkingOutputs\bands\2022_04_25_merged.tif'

def join_rasters(path, output):
    '''
    merges all tif rasters in a folder (path) and saves as a tif as output
    output format needs to be str.tif
    path needs to be a real path r'//'
    requires gdal, subprocess, glob and os
    '''
    ls =  glob.glob(os.path.join(path, '*.tif'))
    vrt = gdal.BuildVRT(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\Permafrost_Sweden\WorkingOutputs\temp.vrt", ls)
    print('starting')
    gdal.Translate(output, vrt)
    vrt = None
    print('finished')

# join_rasters(path, output)
# THIS ALSO DOES NOT WORK BECUASE THE MERGE IS BASED AROUND A SPATIAL MERGE


