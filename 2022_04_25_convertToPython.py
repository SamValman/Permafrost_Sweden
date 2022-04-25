# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 13:31:30 2022
%matplotlib qt
Much of this is copied/altered from Martha Ledger
@author: lgxsv2 samuel.valman@nottingham.ac.uk

# For tensorboard: anaconda: cd to 00:code
# tensorboard --logdir=logs/

"""
#packages#
# general
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
import skimage.io as IO

#ANN##
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import  Dense

# tb
from tensorflow.keras.callbacks import TensorBoard
import time 
#%%
Name = "20mANNchecknormalYlong{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir=f"logs/{Name}")
#%%



fn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\2022_03_martha\NN_dataset.csv"
ds = pd.read_csv(fn)

ds = ds.iloc[:,3:] # remove ID columns 






y =  np.array(ds.iloc[:,9:])
X = np.array(ds.iloc[:,:9])
#try this 
# X = np.array(ds.iloc[:,:4])

''' 
Need to do the normalisation of bands 
# keras.utils.normalise() 
'''
X_normal = keras.utils.normalize(X)
y =  keras.utils.normalize(y)


# needed argmax Labels 
ls = []
for i in y:
    l = np.argmax(i)
    ls.append(l)

y = np.array(ls)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(X_normal, y, test_size=0.2, random_state=42)
# X_train, y_train = tf.convert_to_tensor(X_train), tf.convert_to_tensor(y_train)
# y_train = np.argmax(y_train)





#%%


## Model
# sequential feed forward ANN
model = keras.models.Sequential()

# model.add(keras.layers.Flatten())
model.add(Dense(64, activation=('relu')))
model.add(Dense(64, activation=('relu')))
model.add(Dense(64, activation=('relu')))
#output
model.add(Dense(15, activation=('softmax')))

#compiler
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',
              metrics = ['sparse_categorical_accuracy'])
# print(model.summary())


'''
Training our model
'''

model.fit(X_train, y_train, epochs =50, batch_size=2, callbacks=[tensorboard])


#%%
'''
prediction/test section
'''

test_loss, test_accuracy = model.evaluate(X_test, y_test,callbacks=[tensorboard])
print ('test loss: ', test_loss)

print ('test accuracy:', test_accuracy)
# import cv2
# s2 = cv2.imread(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\2022_03_martha\s2\L1C_T34WDA_A012475_20190727T104030.tif")
# prediction = model.predict(s2)

#%%
## Normalise 
## tensorboard
## do we need f2
## can we run a s2 image? 

#%%


# fn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\s2_files\2019_07_27\T34WDA_20190727T104029_B04_20m.jp2"


# def satfig(fn):
#     im = np.int16(IO.imread(fn))
#     im = keras.utils.normalize(im)
    
#     plt.figure()
#     plt.title('titles')
#     plt.imshow(im) #  clim=(0,0.3)
#     plt.show()  

# satfig(fn)
#%%



fn = "C:\\Users\\lgxsv2\\OneDrive - The University of Nottingham\\PhD\\yr_2\\01_RA2021_2022\\2022_03_arctic\\s2_files\\2019_07_27"



def join_rasters(path):
    '''
    merges all jp2 images in a folder (path) and outputs a multibandArray
    path needs to be a real path r'//' 
    requires , subprocess, numpy as np, skimage.io as IO, and os
    '''
    ls =  glob.glob(os.path.join(path, '*.jp2'))
    im_ls = []
    for filename in ls:
        im = np.int16(IO.imread(filename))
        im = im.flatten()
        im_ls.append(im)
        
    multiBandImage = np.stack(im_ls)
    print(multiBandImage.shape)
    return multiBandImage



s2_2019_07_27 = join_rasters(fn)

X = np.reshape(s2_2019_07_27, (30140100, 9))


#%%
X_normal = keras.utils.normalize(X)

prediction = model.predict(X_normal, verbose=1, batch_size=5)

#%%
print(prediction.shape)

ls = []
tick =1
for i in prediction:
    ti = np.argmax(i)
    tick+=1 
    if tick%1000 == 0:
        print(tick)
    if type(ti) != np.int64:
        print(ti)
        break
    ls.append(ti)
ls = np.array(ls)
#%%
print(ls.shape)
predicted_image = ls.reshape((1,5490, 5490))

#%%
plt.figure()
plt.title('titles')
plt.imshow(predicted_image) #  clim=(0,0.3)
plt.show() 
#%%
# 2022-04-21 14:11:33.979045: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 1808406000 exceeds 10% of free system memory.




fn = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\00_code\outputs\2022_04_21_s2this.tif'
IO.imsave(fn, ls)
#%%
fn = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\00_code\outputs\2022_04_21_ls.npy'
np.save(fn, ls)
fn = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\00_code\outputs\2022_04_21_predicted_image.npy'
np.save(fn, predicted_image)

fn = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\00_code\outputs\2022_04_21_prediction.npy'
np.save(fn, prediction)
#%%
b = np.load(r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\00_code\outputs\2022_04_21_ls.npy")
