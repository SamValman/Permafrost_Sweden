# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 09:49:02 2022

@author: lgxsv2
"""

import pandas as pd
import numpy as np

# fn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2022_10_25_final\globalstatistics\GS.csv"

# df = pd.read_csv(fn)


# mean_percent = (df['PALS'].mean())

# meanHA = (df['PALS_HA'].mean())
# sumHA = (df['PALS_HA'].sum())

# p25 = np.percentile((df['PALS_HA']), 25)
# p75 = np.percentile((df['PALS_HA']), 75)

# print(df['PALS'].mean())

def msq(df):
    a = 0
    for i in df: 
        t = abs(i)
        a = a+t
    return a

fn= r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\summaryList1_nowF1and2\unbuffered_Palsa_tiff.tif"

import skimage.io as IO
im = IO.imread(fn)



im = im[im>-900]

minimum = im.min()
maximum = im.max()
mean = im.mean()
median = np.median(im)
ranges = maximum-minimum

p25 = np.percentile(im, 25)
p75 = np.percentile(im, 75)
total = sum(im)
SquaredTotal = msq(im)

df = pd.DataFrame({'min':[minimum], 'max':[maximum], 'mean':[mean], 'median':[median],
                   'range':[ranges], 'p25': [p25], 'p75':[p75],
                   'total':[total], 'MSTotal':[SquaredTotal]})
df.to_csv(r)























