# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 15:23:28 2022

@author: lgxsv2
"""

import pandas as pd 
import glob
import skimage.io as IO
import matplotlib.pyplot as plt 
import numpy as np

fn = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2022_11_18_final\2022_11_23_Table2\files\*.tif'
fn = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2022_11_18_final\table2\2022_11_14_Table2\*tif'
def giveMeVals(fn):
    print('  ')
    print('  ')
    print('  ')
    print('  ')

    print(fn.split('\\')[-1])

    im = IO.imread(fn)
    im = im[im>-900]
    im = im*1000
    
    total_area = len(im)
    print('total_area ', ((total_area*20)/10000))
    
    negative = im[im<0]
    negative = len(negative)
    print('below zero ', ((negative*20)/10000))
    
    print('min ', im.min())
    print('max ', im.max())
    
    lots = im[im<-3.5]
    print('subsiding most ', ((len(lots)*20)/10000))
    
    
    # minimum = im.min()
    # maximum = im.max()
    # mean = im.mean()
    # median = np.median(im)
    # ranges = maximum-minimum

    # p25 = np.percentile(im, 25)
    # p75 = np.percentile(im, 75)
    
    # top = im[im<-3.5]
    # top = len(top)
    
    return im
    

#%%
for i in glob.glob(fn)[:]:
    # print(i)
    im = giveMeVals(i)


    # break