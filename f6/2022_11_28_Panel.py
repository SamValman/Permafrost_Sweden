# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:05:13 2022

@author: lgxsv2
"""
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from scipy import stats
import os
import matplotlib.gridspec as gridspec
#%%
fn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\f8\rough_InSAR_PP.csv"
df = pd.read_csv(fn)
df1 = df.dropna()

fn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\figure5_Matthias\2022_10_11_extracted.csv"
df0 = pd.read_csv(fn)
#%%#%%

'''
X is PALS bucket
y is radar
color is ProbabilityPermafrost
'''
def fig7Panel(df0,df1=None):
    
    fig, axs = plt.subplots(2,1, figsize=(8,4.5))
    
    # plot scatter
    x1, y1, c1 = df1.Romean, df1['_InSARmean'], df1['_PPmean']
    x0, y0, c0 = df0.PALS, df0['_InSARmean'], df0['_PPmean']
 
    y1, y0 = y1*1000, y0*1000   
 
    cb = axs[1].scatter(x0, y0, c=c0, cmap='magma', alpha=0.7, s=25)
    axs[0].scatter(x1, y1, c=c1, cmap='magma', alpha=0.7, s=25 )
    
    
    # find trendlines
    p0 = trendline(x0, y0)
    p1 = trendline(x1, y1)


    #add trendline to plot
    axs[1].plot(x0, p0(x0), color='black') # , width=2)
    axs[0].plot(x1, p1(x1), color='black') # , width=2)

    #limits
    axs[0].set_xlim(0,1)
    axs[1].set_xlim(0,100)
    axs[1].set_ylim(-7.5, 5)
    axs[0].set_ylim(-7.5,5)




    # labels
    fig.add_subplot(111, frameon=False)
    plt.ylabel('InSAR measured ground motion (mm yr$^-$$^1$)',labelpad=35)
    plt.xticks([], [])
    plt.yticks([], [])

    axs[1].set_xlabel('Percentage palsa ')
    axs[0].set_xlabel('Roughness Index')
    plt.tight_layout()
    
    cbar = fig.colorbar(cb, ax = axs)
    cbar.set_label('Permafrost Probability')
    
    plt.text(0.0,1.01, 'a)')
    plt.text(0.0,0.43, 'b)')

    rsq0, pval0 = stats.pearsonr(x0,y0)
    rsq1, pval1 = stats.pearsonr(x1,y1)
    
    plt.show()
    
    # return rsq0, pval0, rsq1, pval1






def trendline(x, y):
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    return p



fig7Panel(df0, df1)
#%%
fn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\2023_03_03_final\f8\2024_01_10_F6.jpeg"
plt.savefig(fn,dpi=600)
# rsq0, pval0, rsq1, pval1 = 














































# #%%
# old fig scatter

# def figscatter(x,y, inc_trendline=False, x_label='', y_label='', output=None):
#     plt.figure()

    
#     plt.scatter(x, y)

#     z, pval, rsq =None, None, None

#     if inc_trendline:
#         z = np.polyfit(x, y, 1)
#         p = np.poly1d(z)
#         #add trendline to plot
#         plt.plot(x, p(x), color='black') # , width=2)
        
#         # r^2
#         rsq, pval = stats.pearsonr(x,y)
        
#         # put in box 

#         textstr = '\n'.join((
#     r'$Pearson-r=%.2f$' % (rsq, ),
#     r'$p-value=%.2f$' % (pval, )))
        
#         props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#         upper = y.max()-0.001
#         downer = x.max()-0.01
#         plt.text(downer, upper, textstr, fontsize=14,
#         verticalalignment='top', bbox=props)
        
        



    
    
    
#     # Labels
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     # plt.xlim([0,100])
#     # cbar = plt.colorbar()
#     # cbar.set_label('Permafrost Probability')
    
#     plt.show()
    
#     plt.savefig(output, dpi=500)
#     return rsq, pval