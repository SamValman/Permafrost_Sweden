# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:57:04 2022

@author: lgxsv2
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from scipy import stats

#%%

fn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\figure5_Matthias\2022_10_11_extracted.csv"
df = pd.read_csv(fn)
#%%
'''
X is PALS bucket
y is radar
color is ProbabilityPermafrost
'''

def fig5(df, inc_trendline=False):
    plt.figure()
    x = df.PALS
    y = df['_InSARmean']
    
    plt.scatter(x, y, c = df['_PPmean'], cmap='magma')



    if inc_trendline:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        #add trendline to plot
        plt.plot(x, p(x), color='black') # , width=2)
        
        # r^2
        rsq, pval = stats.pearsonr(x,y)
        
        # put in box 

        textstr = '\n'.join((
    r'$Pearson-r=%.2f$' % (rsq, ),
    r'$p-value=%.2f$' % (pval, )))
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        upper = y.max()-0.001
        plt.text(75, upper, textstr, fontsize=14,
        verticalalignment='top', bbox=props)
        
        



    
    
    
    # Labels
    plt.xlabel('Palsa percentage')
    plt.ylabel('InSAR subsidence')
    plt.xlim([0,100])
    cbar = plt.colorbar()
    cbar.set_label('Permafrost Probability')
    
    plt.show()
    return z



#%%

r = fig5(df,inc_trendline=True )
#%%
no_zero = df[df['PALS']!=0]
# make one with removed 0 values 
fig5(no_zero,inc_trendline=True )
#%%

groupeddf = df.groupby(['PALS']).mean().reset_index()

fig5(groupeddf, inc_trendline=True)















