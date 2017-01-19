# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:09:07 2017

@author: elijahc
"""

#%%
import functools
import scipy.io as sio
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import numpy as np
import pyspike as spk

class Model:
    
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        
        hidden_size = 10
        data_size = int(self.data.get_shape()[1])
        batch_size = self.data.get_shape()[0]
        target_size = int(self.target.get_shape()[1]) 
        weight = tf.Variable(tf.truncated_normal([hidden_size, target_size]))
        bias = tf.Variable(tf.constant(0.1, shape=[target_size]))
        
        cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
        vals, states = tf.nn.dynamic_rnn(cell, self.data, dtype=tf.float32)
        
        vals = tf.transpose(vals, [1, 0, 2])
        last = tf.gather_nd(vals, [[int(vals.get_shape()[0])-1]])
        last = tf.squeeze(last)
        
        incoming = tf.matmul(last,weight) + bias
        self._prediction = tf.nn.relu(incoming)
        cm = tf.nn.sigmoid_cross_entropy_with_logits(self._prediction, self.target)
        self._optimize = tf.train.RMSPropOptimizer(0.03).minimize(cm)
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self._prediction, 1))
        self._error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
    
    def launchG(self, fetches, feed_dict):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            vals = sess.run(fetches, feed_dict)
        
        return vals
    
    @property
    def graph(self):
        return self._graph        
            
    @property
    def prediction(self):
        return self._prediction
    
    @property
    def optimize(self):
        return self._optimize
     
    @property
    def error(self):
        return self._error
    
    @property
    def spike_trains(self):
        zero = tf.constant(0, dtype=tf.float32)
        #with tf.device("/cpu:0"):
        t = tf.transpose(tf.where(tf.not_equal(self.target,zero)), [1,0])
        g = tf.transpose(tf.where(tf.not_equal(self._prediction,zero)), [1,0])
        
        return t,g

#%%        
def main():
#%%
    # load data

    NUM_EXAMPLES = 2000
    mat_file = sio.loadmat('train.mat')
    inputs = np.expand_dims(mat_file['stim_grate'], axis=2)
    targets = mat_file['resp_grate']
    test_input = inputs[NUM_EXAMPLES:]
    test_output = targets[NUM_EXAMPLES:]
    
    train_input = inputs[:NUM_EXAMPLES]
    train_output = targets[:NUM_EXAMPLES]
    #import pdb; pdb.set_trace()

    # Create model
    epoch = 13
    batch_size = np.shape(targets)[2]
    stim_t = tf.placeholder(tf.float32, [batch_size,267,1])
    response_t = tf.placeholder(tf.float32, [batch_size, 266])

    train_output = np.transpose(train_output, [0,2,1])
    train_input = np.transpose(train_input, [2,1,0])
    train_input = np.expand_dims(train_input, axis=2)

    with tf.name_scope('Train'):
        with tf.variable_scope('Model', reuse=None):
            model = Model(stim_t, response_t)
 
    for i in range(epoch):
        for j in range(NUM_EXAMPLES):
            inp,out = np.repeat(train_input[:,:,:,j], batch_size, axis=0), train_output[j]
            inp = np.transpose(inp, [2,1,0])
            inp = np.insert(inp, 0, np.arange(batch_size), axis=1)
            inp = np.transpose(inp, [2,1,0])
            
            fetches = {'eval':model.optimize,
                       'error':model.error,
                       'spike_trains': model.spike_trains
                       }
            feed_dict = {stim_t: inp, response_t:out}
            vals = model.launchG(fetches, feed_dict)
            if j % 10 == 0:
                print("Epoch: %d Example: %d Error: %.3f" % (i+1,j+1, 100*vals['error']))
                print(vals['spike_trains'])
                #summary_writer.add_summary(vals['summary'],model.global_step.eval(session=sess))
        incorrect = sess.run(model.error,{stim_t: test_input, response_t:test_output})
        print("Epoch: %d Error: %.3f" % (i + 1, 100 * incorrect))

        
if __name__ == '__main__':
    main()