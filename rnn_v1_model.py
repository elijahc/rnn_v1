# -*- coding: utf-8 -*-

import scipy.io as sio
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np


class Model:

    def __init__(self, x, y, num_steps=5, state_size=4, learning_rate=1e-4):
        # Config Variables
        self.x = x
        self.y = y
        self.num_steps = num_steps  # number of truncated backprop steps ('n')
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        num_classes = 2  # num of possible output (0,1) in this case.
        num_layers = 2

        batch_size = self.x.get_shape()[0]

        embeddings = tf.get_variable('embedding_matrix',
                                     [num_classes, state_size])

        rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

        weight = tf.Variable(tf.truncated_normal([state_size, num_classes]))
        bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))

        init_state = tf.zeros([batch_size, state_size])
        cell = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)
        #cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers,state_is_tuple=True)

        rnn_outputs, final_state = tf.nn.dynamic_rnn(
            cell,
            rnn_inputs,
            initial_state=init_state,
            dtype=tf.float32
        )

        rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
        last = tf.gather_nd(rnn_outputs, [[int(rnn_outputs.get_shape()[0])-1]])
        last = tf.squeeze(last)

        incoming = tf.matmul(rnn_outputs, weight) + bias
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
        t = tf.where(tf.not_equal(self.target,zero))
        g = tf.where(tf.not_equal(self._prediction,zero))

        return t,g

def munge(data, targets, i):
    inp,out = np.repeat(data[:,:,:,j], batch_size, axis=0), targets[i]
    inp = np.transpose(inp, [2,1,0])
    inp = np.insert(inp, 0, np.arange(batch_size), axis=1)
    inp = np.transpose(inp, [2,1,0])

    return {'stim_t': inp, 'response_t':out}


def train(session):
    for i in range(epoch):
        for j in range(NUM_EXAMPLES):
            feed_dict = munge(train_input, train_output, j)
            fetches = {'eval':model.optimize,
                    'error':model.error,
                    'spike_trains': model.spike_trains
                    }
            vals = model.launchG(fetches, feed_dict)
            if j % 10 == 0:
                print("Epoch: %d Example: %d Error: %.3f" % (i+1,j+1, 100*vals['error']))
                print(vals['spike_trains'])
                #summary_writer.add_summary(vals['summary'],model.global_step.eval(session=sess))
        incorrect = sess.run(model.error,{stim_t: test_input, response_t:test_output})
        print("Epoch: %d Error: %.3f" % (i + 1, 100 * incorrect))

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


    # Set params/placeholders
    epoch = 13
    batch_size = 200
    num_steps = 5
    x = tf.placeholder(tf.int32, [batch_size,num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    train_output = np.transpose(train_output, [0,2,1])
    train_input = np.transpose(train_input, [2,1,0])
    train_input = np.expand_dims(train_input, axis=2)

    with tf.name_scope('Train'):
        with tf.variable_scope('Model', reuse=None):
            model = Model(stim_t, response_t)
    feed_dict = munge(train_input, train_output, 20)
    fetchers = {
            'spikes': spike_trains
            }

    vals = model.launchG(fetchers, feed_dict)

if __name__ == '__main__':
    main()
