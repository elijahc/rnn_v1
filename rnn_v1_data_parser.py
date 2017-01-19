# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:06:34 2017

@author: elijahc
"""

#%%
import tensorflow as tf
import scipy.io as sio
import numpy as np
from tqdm import tqdm

train_data = sio.loadmat('train.mat')
#%%

sequences = train_data['stim_grate']
labels = train_data['resp_grate']
labels = np.swapaxes(labels,1,2)

#%%

def _int64_feature(value):
    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    
def _make_sequence_example(sequence, labels):
    ex = tf.train.SequenceExample()
             
    sequence_length = len(sequence)
    num_neurons = np.size(labels, axis=1)
    
    ex.context.feature['length'].int64_list.value.append(sequence_length)
    ex.context.feature['num_neurons'].int64_list.value.append(num_neurons)                  
    fl_stim = ex.feature_lists.feature_list["image_stim"]
    fl_spikes = ex.feature_lists.feature_list['spikes']
    
    for stim, spikes in zip(sequence.tolist(),labels.tolist()):
        fl_stim.feature.add().int64_list.value.append(stim)
        fl_spikes.feature.add().bytes_list.value.append(str.encode(''.join(map(str,spikes)))) 

    return ex

def save_tf(FILENAME, sequences, labels):
    with open(FILENAME, 'w') as fp:
        writer = tf.python_io.TFRecordWriter(fp.name)
        for sequence, label_sequence in tqdm(zip(sequences, labels)):
            ex = _make_sequence_example(sequence, label_sequence)
            writer.write(ex.SerializeToString())
        writer.close()
        
save_tf('train.tfrecords', sequences, labels)