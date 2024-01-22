# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 03:03:02 2023

@author: lgxsv2
"""

import skimage.io as IO
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
#%%
fp = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\f5\Verror\*.tif"
def getMSE(fn):
    
    im = IO.imread(fn)
    im = im[im>0]
    print(fn.split('\\')[-1])

    print(np.mean(im))
    print('max: ', im.max())
    
    
for i in glob.glob(fp):
    
    getMSE(i)
    
#%% For the text we also explain how much of the data is likely to change direction as a result of the MSE. 
fn_error = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\f5\Verror\*.tif"
fn_im = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham/PhD/yr_2/01_RA2021_2022/2022_03_arctic/.FinalFigures/summaryList1_nowF1and2/bufferedPalsa_tif/*.tif"

def percentDirectionChange(fn_im, fn_error): 

    print(fn_error.split('\\')[-1])
    er = IO.imread(fn_error)
    er = er[er>0]
    er = np.mean(er)
    
    
    im = IO.imread(fn_im)
    total = np.sum(im>-900)

    im = im[im>-900]

    im_pos = im[im>0]
    im_pos = im_pos-er
    cp = np.sum(im_pos<0)
    
    im_neg = im[im<0]
    im_neg = im_neg - er
    cn = np.sum(im_neg>0)

    count = cn + cp
    

    return count, total


#%%
counts = 0
totals = 0
#%%
for i, l in zip(sorted(glob.glob(fn_im)), sorted(glob.glob(fn_error))):
    c, t = percentDirectionChange(i, l)
    counts += c
    totals += t
    
#%%

# Some stats with the above to get total percentage.
percent = (counts/totals)*100
    