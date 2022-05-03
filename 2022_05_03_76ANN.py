# -*- coding: utf-8 -*-
"""
###############################################################################
Project: SWEDISH PERMAFROST
###############################################################################
This was found to have a 76.1% accuracy on the FCM 

THIS IS A SCRIPT JUST TO TRAIN AND SAVE THE MODEL.

Created on Thu Apr 28 10:41:01 2022

@author: lgxsv2
"""
#%% Packages
print('''
      Two sections of script:
          - training and testing
          - Prediction 
          packages only needed for section 2 are loaded there. 
      ''')
# general
import pandas as pd
import numpy as np
import time 


#ML/AI/Stats packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import  Dense
from sklearn.model_selection import train_test_split


# For tensorboard: anaconda: cd to 00:code
# tensorboard --logdir=logs
from tensorflow.keras.callbacks import TensorBoard

#%% Data formatting

fn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\2022_03_martha\NN_dataset.csv"
ds = pd.read_csv(fn)

ds = ds.iloc[:,3:] # remove ID columns 

#predictor/labels split
y =  np.array(ds.iloc[:,9:]) 
X = np.array(ds.iloc[:,:9])

#Normalise(normalize)
X_normal = keras.utils.normalize(X)
y =  keras.utils.normalize(y)

X_train, X_test, y_train, y_test = train_test_split(X_normal, y, test_size=0.2, random_state=42)
#%%Here lies the issue
print('''
      The training dataset is heavily biased towards landcover 0
      we can see this if we argmax the set
      ''')
     
def argmax_a_prediction(prediction):
    '''
    name is a bit misleading
    will argmax any dataset/array

    '''
    ls = []
    for i in prediction:
        temp = np.argmax(i)
        ls.append(temp)
        #in case of an issue
        if type(temp)!=np.int64:
            print(temp)
            break
    return  np.array(ls)
y_argmax = argmax_a_prediction(y)
for i in range(15):
    print(y_argmax[y_argmax==i].shape)

#%% Model Parameters
'''
LIST OF ATTEMPTS (however stupid) in github repo under "2022_04_28_modelRuns.xlsx"
'''
# final layer activation function - this is the output they want 
final_layer_AF = 'softmax'
#compiler
optim = 'adam'
loss = 'CategoricalCrossentropy'
metric = 'Accuracy'

epochs=20

#%% Model
#TB organising
Name = "20mANN{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir=f"logs/{Name}")

# sequential feed forward ANN
model = keras.models.Sequential()

model.add(Dense(64, activation=('relu')))
# model.add(Dense(64, activation=('relu')))
# model.add(Dense(64, activation=('relu')))
#output
model.add(Dense(15, activation=(final_layer_AF)))

#compiler
model.compile(optimizer = optim, loss = loss, metrics = [metric])

#fit model
model.fit(X_train, y_train, epochs=epochs, batch_size=64, callbacks=[tensorboard])

# results
test_loss, test_accuracy = model.evaluate(X_test, y_test,callbacks=[tensorboard])
print ('test loss: ', test_loss)
print ('test accuracy:', test_accuracy)
#%%
# Predict the test set 
internal_predict = model.predict(X_test, verbose=1, batch_size=6)
print(internal_predict)
#argmax result
internal_argmax = argmax_a_prediction(internal_predict)
print(np.unique(internal_argmax))
#%% Format for Confusion Matrix
t = pd.DataFrame(internal_predict)
fn = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\Permafrost_Sweden\WorkingOutputs\2022_04_29.csv'
t.to_csv(fn)

t = pd.DataFrame(y_test)
fn = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\Permafrost_Sweden\WorkingOutputs\2022_05_03_Y.csv'
t.to_csv(fn)
#%% Save the model

# saving without a file extension creates a recommended TF SavedModel format rather than h5
model.save('MR1_76')
# check the save 
model_fn = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\Permafrost_Sweden\MR1_76'
m = keras.models.load_model(model_fn)