# -*- coding: utf-8 -*-
"""
Created on Fri May 26 16:19:47 2023

@author: lgxsv2
"""

import skimage.io as IO
import numpy as np
import cv2


fn_wm = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\summaryList1_nowF1and2\unbuffered_Palsa_tiff.tif"

def remove_specal(im, kernel_size=3):
    '''
    uses morphological operators from image processing orginally from another of my phd papers. 
    Parameters
    ----------
    im : array
        P2 out put.
    kernel_size : int, optional
        kernal for operator. The default is 3.

    Returns
    -------
    output : TYPE
        DESCRIPTION.

    '''
    #reverse image because dilation works on one class primarily 
    # our method seems to bleed land into water therefore this is the way we want it 
    im_reversed = 1-im 
    
    # needs to be this format for cv2
    input_im = im_reversed.astype('uint8')
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilution = cv2.dilate(input_im, kernel)
    
    #revert to normal 
    output = 1-dilution
    return output





#%%
#requirements  
import shutil
import glob
import os
import json
import pandas as pd
import numpy as np
import rasterio as rio

# to save with the geo info:::::::
    
with rio.open(fn_wm) as m:
     im = m.read()
     profile = m.profile
 
 
out = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\f1\newPalsaTif.tif'
 #%%
 # return profile
im =im[0]
il = remove_specal(im)

#%%
 # save file with new/old profile
with rio.open(out, 'w', **profile) as l:
 l.write(il, indexes=1)
print('complete')