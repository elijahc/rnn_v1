# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%
import scipy.io as sio
import numpy as np

# Load files
stim_sequence = sio.loadmat('data/stimulus_sequence.mat')['stimulus_sequence']
session_1 = sio.loadmat('data/10.mat')

#%%
# Filter out poor quality neurons
mask = np.squeeze(session_1['INDCENT']).astype(bool)

resp_train = session_1['resp_train'][mask]
resp_train_blk = session_1['resp_train_blk'][mask]

# Shift by 50ms to account for response latency
latency = 50
resp = np.concatenate((resp_train,resp_train_blk), axis=3)
#resp = np.roll(resp,-latency,3)[:,:,:,:-latency]


#%%

# 105+211*956*20
def mutate(resp):
    sequences = []
    labels = []
    trials = np.size(resp,2)
    num_neurons = np.size(resp,0)
    num_images = np.size(resp, 1)
    i = 0
    for r in np.arange(trials):
        for image_id in stim_sequence[:,r]:
            index = {'i': i,
                     'trial': r,
                     'image': image_id-1
                     }
            x_on = np.zeros(105, dtype=int) + image_id
            x_off= np.zeros(211, dtype=int) + 956
            x = np.concatenate((x_on, x_off))
            
            y = resp[:,image_id-1, r,:]
            i = i+1
            sequences.extend([x])
            labels.extend([y])
            #print(index)
            #print(time_pts)
            #print(time_pts+316*i)
            #print(ms)
        print(index)
        print(x.shape)
        print(x)
        print(y.shape)
        print(y)
    return (sequences,np.concatenate(np.array(labels), axis=1))
seq, lab = mutate(resp)

sio.savemat('data/10_timeseries.mat', {'timeseries':lab})
