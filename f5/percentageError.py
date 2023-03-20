# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 15:51:10 2023

@author: lgxsv2
"""

import skimage.io as IO
import glob 
import numpy as np
import pandas as pd

fn = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2022_11_18_final\f2\Verror\*.tif'

b = np.array([])
for i in glob.glob(fn):
    
    im = IO.imread(i)
    im = im.flatten()
    b = np.concatenate([b,im])
blen = b.shape[0]
b=b*1000

numOver1mm = b[b<=1.52].shape[0]

p = numOver1mm/blen
p = p*100

print(p)


 
