# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:11:38 2022

@author: lgxsv2
"""

import pandas as pd
import skimage.io as IO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


xys = pd.read_csv(r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\f4\vectors\xys.csv')
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

plt.close('all')
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

axs[0][0].text(.01, .98, ('('+str(xys.iloc[6,1])[:5]+', '+str(xys.iloc[6,0])[:5])+')',ha='left', va='top', transform=axs[0][0].transAxes)
axs[0][1].text(.01, .98, ('('+str(xys.iloc[3,1])[:5]+', '+str(xys.iloc[3,0])[:5])+')',ha='left', va='top', transform=axs[0][1].transAxes)
axs[1][0].text(.01, .98, ('('+str(xys.iloc[2,1])[:5]+', '+str(xys.iloc[2,0])[:5])+')',ha='left', va='top', transform=axs[1][0].transAxes)
axs[1][1].text(.01, .98, ('('+str(xys.iloc[0,1])[:5]+', '+str(xys.iloc[0,0])[:5])+')',ha='left', va='top', transform=axs[1][1].transAxes)
axs[2][0].text(.01, .98, ('('+str(xys.iloc[1,1])[:5]+', '+str(xys.iloc[1,0])[:5])+')',ha='left', va='top', transform=axs[2][0].transAxes)
axs[2][1].text(.01, .98, ('('+str(xys.iloc[7,1])[:5]+', '+str(xys.iloc[7,0])[:5])+')',ha='left', va='top', transform=axs[2][1].transAxes)
axs[3][0].text(.01, .98, ('('+str(xys.iloc[5,1])[:5]+', '+str(xys.iloc[5,0])[:5])+')',ha='left', va='top', transform=axs[3][0].transAxes)
axs[3][1].text(.01, .98, ('('+str(xys.iloc[4,1])[:5]+', '+str(xys.iloc[4,0])[:5])+')',ha='left', va='top', transform=axs[3][1].transAxes)
# axs[0][1].set_title(r'Viss$\acute{a}$tvuopmi')
# axs[1][0].set_title('Western Tavvavuoma')
# axs[1][1].set_title('Tavvavuoma')
# axs[2][0].set_title('Ragesvuomus-Pirttimysvuoma')
# axs[2][1].set_title('Ribasvuomus')
# axs[3][0].set_title('Gipmevuopmi')
# axs[3][1].set_title('Sirccam')


plt.xticks([], [])
plt.yticks([], [])


plt.ylabel("Frequency of Pixels", fontsize=14, labelpad=35)
plt.xlabel("Ground motion (mm yr$^-$$^1$)", fontsize=14, labelpad=15)

fn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\f4\2024_01_18_F3First.jpeg"
plt.savefig(fn,dpi=600, bbox_inches='tight')

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

fn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\f4\2023_10_10_F3Second.jpeg"
plt.savefig(fn,dpi=600)
# highlight 0 line 
# shade below 0 line









#%%





fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(5, 2, height_ratios=[1, 1, 1, 1, 2])

###################### top
ax0 = plt.subplot(gs[0, :])  # Span all columns in the first row
fn_all = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\summaryList1_nowF1and2\unbuffered_Palsa_tiff.tif"

al = sort(fn_all)


ax0.axvline(0, color='red', linestyle='dashed', linewidth=2)
mask = (al < 0.06)
bins = np.linspace(-10,7,50)
ax0.hist(al[mask],bins=bins, histtype='bar',rwidth=0.85, edgecolor='black', facecolor='dimgray')#, lw=0)

ax0.hist(x=al, bins=bins, facecolor="None", rwidth=0.85, edgecolor='black')

ax0.set_title('All palsa complexes')


###### others ###########################
#### bins
bins = np.linspace(-10,7,50)

# number
# naming matches old subplots
axs_00 = plt.subplot(gs[1, 0])  
axs_01 = plt.subplot(gs[1, 1],sharex=axs_00)
axs_10 = plt.subplot(gs[2, 0],sharex=axs_00)
axs_11 = plt.subplot(gs[2, 1],sharex=axs_00)
axs_20 = plt.subplot(gs[3, 0],sharex=axs_00)
axs_21 = plt.subplot(gs[3, 1],sharex=axs_00)
axs_30 = plt.subplot(gs[4, 0],sharex=axs_00)
axs_31 = plt.subplot(gs[4, 1],sharex=axs_00)


# hists
axs_00.hist(x=p4, bins=bins, facecolor="None", rwidth=0.85, edgecolor='black')
mask = (p4 < 0.06)
axs_00.hist(p4[mask],bins=bins, histtype='bar',rwidth=0.85, edgecolor='black', facecolor='dimgray')#, lw=0)


axs_01.hist(x=p1, bins=bins, facecolor="None", rwidth=0.85, edgecolor='black')
mask = (p1 < 0.06)
axs_01.hist(p1[mask],bins=bins, histtype='bar',rwidth=0.85, edgecolor='black', facecolor='dimgray')#, lw=0)

axs_10.hist(x=p3, bins=bins, facecolor="None",  rwidth=0.85, edgecolor='black')
mask = (p3 < 0.06)
axs_10.hist(p3[mask],bins=bins, histtype='bar',rwidth=0.85, edgecolor='black', facecolor='dimgray')#, lw=0)

axs_11.hist(x=p2, bins=bins, facecolor="None",  rwidth=0.85, edgecolor='black')
mask = (p2 < 0.06)
axs_11.hist(p2[mask],bins=bins, histtype='bar',rwidth=0.85, edgecolor='black', facecolor='dimgray')#, lw=0)

axs_20.hist(x=p6, bins=bins, facecolor="None",  rwidth=0.85, edgecolor='black')
mask = (p6 < 0.06)
axs_20.hist(p6[mask],bins=bins, histtype='bar',rwidth=0.85, edgecolor='black', facecolor='dimgray')#, lw=0)

axs_21.hist(x=p8, bins=bins, facecolor="None",  rwidth=0.85, edgecolor='black')
mask = (p8 < 0.06)
axs_21.hist(p8[mask],bins=bins, histtype='bar',rwidth=0.85, edgecolor='black', facecolor='dimgray')#, lw=0)

axs_30.hist(x=p5, bins=bins, facecolor="None",  rwidth=0.85, edgecolor='black')
mask = (p5 < 0.06)
axs_30.hist(p5[mask],bins=bins, histtype='bar',rwidth=0.85, edgecolor='black', facecolor='dimgray')#, lw=0)

axs_31.hist(x=p7, bins=bins, facecolor="None", rwidth=0.85, edgecolor='black')
mask = (p7 < 0.06)
axs_31.hist(p7[mask],bins=bins, histtype='bar',rwidth=0.85, edgecolor='black', facecolor='dimgray')#, lw=0)



# av line
axs_00.axvline(0, color='red', linestyle='dashed', linewidth=2)
axs_01.axvline(0, color='red', linestyle='dashed', linewidth=2)
axs_10.axvline(0, color='red', linestyle='dashed', linewidth=2)
axs_11.axvline(0, color='red', linestyle='dashed', linewidth=2)
axs_20.axvline(0, color='red', linestyle='dashed', linewidth=2)
axs_21.axvline(0, color='red', linestyle='dashed', linewidth=2)
axs_30.axvline(0, color='red', linestyle='dashed', linewidth=2)
axs_31.axvline(0, color='red', linestyle='dashed', linewidth=2)


# titles
axs_00.set_title(r'$\acute{A}$rbuvuopmi')
axs_01.set_title(r'Viss$\acute{a}$tvuopmi')
axs_10.set_title('Western Tavvavuoma')
axs_11.set_title('Tavvavuoma')
axs_20.set_title('Ragesvuomus-Pirttimysvuoma')
axs_21.set_title('Ribasvuomus')
axs_30.set_title('Gipmevuopmi')
axs_31.set_title('Sirccam')

#x an y 


axs_00.set_xticks([-10,-7.5, -5, -2.5, 0, 2.5, 5, 7.5])
axs_01.set_xticks([])
axs_10.set_xticks([])
axs_11.set_xticks([])
axs_20.set_xticks([])
axs_21.set_xticks([])
axs_30.set_xticks([])
axs_31.set_xticks([])
## main labels
fig.add_subplot(111, frameon=False)

plt.xticks([], [])
plt.yticks([], [])


plt.ylabel("Frequency of Pixels", fontsize=14, labelpad=35)
plt.xlabel("Subsidence (mm yr$^-$$^1$)", fontsize=14, labelpad=15)
