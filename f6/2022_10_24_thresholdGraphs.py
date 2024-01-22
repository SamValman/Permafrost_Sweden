# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:57:04 2022

@author: lgxsv2
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from scipy import stats
import os

#%%

fn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\threshold figures\ZS_24_10_completed.csv"
df = pd.read_csv(fn)
df = df.dropna()

#%%
'''
X is PALS bucket
y is radar
color is ProbabilityPermafrost
'''

def figscatter(x,y, inc_trendline=False, x_label='', y_label='', output=None):
    plt.figure()

    
    plt.scatter(x, y)

    z, pval, rsq =None, None, None

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
        downer = x.max()-0.01
        plt.text(downer, upper, textstr, fontsize=14,
        verticalalignment='top', bbox=props)
        
        



    
    
    
    # Labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.xlim([0,100])
    # cbar = plt.colorbar()
    # cbar.set_label('Permafrost Probability')
    
    plt.show()
    
    plt.savefig(output, dpi=500)
    return rsq, pval

#%%
x = df.In_mean
for i in ['In_mean', 'In_max', 'In_min']:
    x = df[i]
    fn = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\.FinalFigures\threshold figures'
    # folder = os.path.join(fn+i)
    # print('f')
    # print(folder)
    # print('f')

    for m in ['ro_median', 'ro_sum', 'ro_mean', 'ro_stdev', 'ro_min', 'ro_max', 'ro_range', 'ro_variance']:
        y = df[m]
        imageName = m+'.png'
        output = os.path.join(*[fn, i, imageName])
        rsq, pval = figscatter(x, y, inc_trendline=True, x_label=i, y_label=m, output=output)
        
        












# r = fig5(df,inc_trendline=True )
# #%%
# no_zero = df[df['PALS']!=0]
# # make one with removed 0 values 
# fig5(no_zero,inc_trendline=True )
# #%%

# groupeddf = df.groupby(['PALS']).mean().reset_index()

# fig5(groupeddf, inc_trendline=True)















