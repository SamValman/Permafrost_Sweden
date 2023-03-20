# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:11:38 2022

@author: lgxsv2
"""

import pandas as pd
import skimage.io as IO
import numpy as np
import matplotlib.pyplot as plt

#%%
import glob
fn = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\summaryList1_nowF1and2\bufferedPalsa_tif\*.tif'
fps = glob.glob(fn)
def sort(fn):
    df = IO.imread(fn)
    print(fn)
    df = df[df>-900]
    df = df / 0.001
    return df

#%%

for x in range(0, 8):
    i = fps[x]
    
    globals()['df%s' % x] = sort(i)
#%%
p8 = df0
p5 = df1
p4 = df2
p1 = df3
p7 = df4
p6 = df5
p3 = df6
p2 = df7


fig, axs = plt.subplots(4,2, sharex=True)
plt.tight_layout()
bins = np.linspace(-10,7,50)


axs[0][0].hist(x=p4, bins=bins, facecolor="None", rwidth=0.85, edgecolor='black')
mask = (p4 < 0.06)
axs[0][0].hist(p4[mask],bins=bins, histtype='bar',rwidth=0.85, edgecolor='black', facecolor='dimgray')#, lw=0)


axs[0][1].hist(x=p1, bins=bins, facecolor="None", rwidth=0.85, edgecolor='black')
mask = (p1 < 0.06)
axs[0][1].hist(p1[mask],bins=bins, histtype='bar',rwidth=0.85, edgecolor='black', facecolor='dimgray')#, lw=0)

axs[1][0].hist(x=p3, bins=bins, facecolor="None",  rwidth=0.85, edgecolor='black')
mask = (p3 < 0.06)
axs[1][0].hist(p3[mask],bins=bins, histtype='bar',rwidth=0.85, edgecolor='black', facecolor='dimgray')#, lw=0)

axs[1][1].hist(x=p2, bins=bins, facecolor="None",  rwidth=0.85, edgecolor='black')
mask = (p2 < 0.06)
axs[1][1].hist(p2[mask],bins=bins, histtype='bar',rwidth=0.85, edgecolor='black', facecolor='dimgray')#, lw=0)

axs[2][0].hist(x=p6, bins=bins, facecolor="None",  rwidth=0.85, edgecolor='black')
mask = (p6 < 0.06)
axs[2][0].hist(p6[mask],bins=bins, histtype='bar',rwidth=0.85, edgecolor='black', facecolor='dimgray')#, lw=0)

axs[2][1].hist(x=p8, bins=bins, facecolor="None",  rwidth=0.85, edgecolor='black')
mask = (p8 < 0.06)
axs[2][1].hist(p8[mask],bins=bins, histtype='bar',rwidth=0.85, edgecolor='black', facecolor='dimgray')#, lw=0)

axs[3][0].hist(x=p5, bins=bins, facecolor="None",  rwidth=0.85, edgecolor='black')
mask = (p5 < 0.06)
axs[3][0].hist(p5[mask],bins=bins, histtype='bar',rwidth=0.85, edgecolor='black', facecolor='dimgray')#, lw=0)

axs[3][1].hist(x=p7, bins=bins, facecolor="None", rwidth=0.85, edgecolor='black')
mask = (p7 < 0.06)
axs[3][1].hist(p7[mask],bins=bins, histtype='bar',rwidth=0.85, edgecolor='black', facecolor='dimgray')#, lw=0)


axs[0][0].axvline(0, color='red', linestyle='dashed', linewidth=2)
axs[0][1].axvline(0, color='red', linestyle='dashed', linewidth=2)
axs[1][0].axvline(0, color='red', linestyle='dashed', linewidth=2)
axs[1][1].axvline(0, color='red', linestyle='dashed', linewidth=2)
axs[2][0].axvline(0, color='red', linestyle='dashed', linewidth=2)
axs[2][1].axvline(0, color='red', linestyle='dashed', linewidth=2)
axs[3][0].axvline(0, color='red', linestyle='dashed', linewidth=2)
axs[3][1].axvline(0, color='red', linestyle='dashed', linewidth=2)

fig.add_subplot(111, frameon=False)

# titles
axs[0][0].set_title(r'$\acute{A}$rbuvuopmi')
axs[0][1].set_title(r'Viss$\acute{a}$tvuopmi')
axs[1][0].set_title('Western Tavvavuoma')
axs[1][1].set_title('Tavvavuoma')
axs[2][0].set_title('Ragesvuomus-Pirttimysvuoma')
axs[2][1].set_title('Ribasvuomus')
axs[3][0].set_title('Gipmevuopmi')
axs[3][1].set_title('Sirccam')




plt.xticks([], [])
plt.yticks([], [])


plt.ylabel("Frequency of Pixels", fontsize=14, labelpad=35)
plt.xlabel("Subsidence (mm)", fontsize=14, labelpad=15)

#%%

fn_all = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\summaryList1_nowF1and2\unbuffered_Palsa_tiff.tif"

al = sort(fn_all)


plt.figure(figsize=(10,2))

plt.axvline(0, color='red', linestyle='dashed', linewidth=2)
mask = (al < 0.06)
bins = np.linspace(-10,7,50)
plt.hist(al[mask],bins=bins, histtype='bar',rwidth=0.85, edgecolor='black', facecolor='dimgray')#, lw=0)

plt.hist(x=al, bins=bins, facecolor="None", rwidth=0.85, edgecolor='black')

plt.title('All palsa complexes')
plt.show()

# highlight 0 line 
# shade below 0 line






















