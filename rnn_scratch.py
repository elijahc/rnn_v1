# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 16:47:50 2017

@author: elijahc
"""

import numpy as np
import tensorflow as tf

num_steps = 5
batch_size = 200
num_classes = 2
state_size = 4
learning_rate = 0.1

"""
Placeholders
"""

x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')
init_state = tf.zeros([batch_size, state_size])

"""
RNN Inputs
"""

# Turn our x placeholder into a list of one-hot tensors:
# rnn_inputs is a list of num_steps tensors with shape [batch_size, num_classes]
x_one_hot = tf.one_hot(x, num_classes)
rnn_inputs = tf.unpack(x_one_hot, axis=1)