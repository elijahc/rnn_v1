# -*- coding: utf-8 -*-

import scipy.io as sio
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import time
from tqdm import tqdm


class Model:

    def __init__(self, x, y, x_mean, num_steps=50, state_size=25, learning_rate=1e-3, num_layers=3):

        # Config Variables
        self.x = x # [1,num_steps,n_use=state_size]
        self.y = y
        self.num_steps = num_steps  # number of truncated backprop steps ('n')
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        global_step = tf.Variable(0, name='global_step', trainable=False)
        batch_size = self.x.get_shape()[0]
        self.n_use = int(self.x.get_shape()[2])

        rnn_inputs = self.x
        weight = tf.get_variable('weight',[self.state_size, self.n_use], initializer=tf.constant_initializer(1.0))
        bias = tf.get_variable('bias', [self.n_use], initializer=tf.constant_initializer(0.1))

        cell = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers,state_is_tuple=True)
        self.init_state = cell.zero_state(batch_size, tf.float32)

        # rnn_outputs shape = (batch_size, num_steps, state_size)
        # final_state = rnn_outputs[:,-1,:] = (30, 25)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(
            cell,
            rnn_inputs,
            initial_state=self.init_state,
            dtype=tf.float32
        )
        # Flatten rnn_outputs down to shape = (batch_size*num_steps, state_size)
        rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
        # Flatten y; (num_steps,n_use) -> (num_steps*n_use)
        y_reshaped = tf.reshape(self.y, [-1, self.n_use])

        # (1500,25) x (25,2) = (1500,2)
        logits = tf.matmul(rnn_outputs, weight) + bias
        #logits = tf.reshape(logits, [int(batch_size),self.num_steps])
        self.debug=logits
        #seqw =  tf.ones((batch_size, num_steps))
        self._prediction = logits
        #self._prediction = tf.reshape(self._flat_prediction, [-1, self.num_steps])
        #for i in range(self.n_use-1):
            #tf.summary.histogram('prediction_n%d'%(i),self._prediction[i,:])

        with tf.name_scope('error'):
            self._error = logits-self.y
            with tf.name_scope('total_loss'):
                self._total_loss = tf.reduce_sum(tf.pow(self._error,2))

        #logits_1 = tf.reshape(logits, [-1, num_steps, num_classes])
        #seq_loss = tf.nn.seq2seq.sequence_loss_by_example(
        #    tf.unpack(logits_1, axis=1),
        #    tf.unpack(self.y, axis=1),
        #    tf.unpack(seqw, axis=1),
        #    average_across_timesteps=True
        #    #softmax_loss_function=
        #)
        #perplexity = tf.exp(seq_loss)
        #import pdb; pdb.set_trace()
        #self._avg_perplexity = tf.reduce_mean(perplexity)
        tf.summary.scalar('total_loss', self._total_loss)
        #tf.summary.scalar('avg_perplexity', self._avg_perplexity)
        self._optimize = tf.train.AdamOptimizer(learning_rate).minimize(self._total_loss, global_step=global_step)

        with tf.name_scope('accuracy'):
             with tf.name_scope('correct_prediction'):
                #correct_prediction = self.prediction-self.y
                null_error = tf.abs(tf.zeros_like(self.prediction)-self.y)
                mean_error = tf.abs(x_mean-self.y)
             #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
             null_loss = tf.reduce_sum(tf.pow(null_error, 2))
             var = tf.reduce_sum(tf.pow(mean_error, 2))
             self.FEV = 1-(self._total_loss/var)
        #tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('null_loss', null_loss)
        tf.summary.scalar('var', var)
        tf.summary.scalar('rel_mean', var/self._total_loss)
        tf.summary.scalar('rel_null', null_loss/self._total_loss)
        tf.summary.scalar('FEV', 1-(self._total_loss/var))
        #tf.summary.scalar('relative_accuracy', accuracy/null_accuracy)

        self._merge_summaries = tf.summary.merge_all()
        self._global_step = global_step

    def do(self, session, fetches, feed_dict):
        vals = session.run(fetches, feed_dict)

        return vals

    def step(self,session):
        return tf.train.global_step(session,self._global_step)

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
    def avg_perplexity(self):
        return self._avg_perplexity

    @property
    def total_loss(self):
        return self._total_loss
    @property
    def merge_summaries(self):
        return self._merge_summaries

def gen_batch(raw_data, batch_size, num_steps, idxs):
    raw_x, raw_y = raw_data
    data_length = np.size(raw_x,axis=-1)
    #raw_x = np.squeeze(raw_x)
    #raw_y = np.squeeze(raw_y)

    # partition raw data into batches and stack them vertically in a data matrix
    #batch_partition_length = data_length // batch_size
    #data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    #data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    #for i in range(batch_size):
    #    data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
    #    data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]

    # further divide batch partitions into num_steps for truncated backprop
    data_x, data_y = raw_x, raw_y
    epoch_size = data_length // num_steps

    for i in idxs:
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        #import pdb;pdb.set_trace()
        yield (x, y)

def gen_epochs(raw_data,n, batch_size,num_steps,idxs):
    for i in range(n):
        yield gen_batch(raw_data, batch_size, num_steps,idxs)

def main():

    # Set params/placeholders
    tf.reset_default_graph()
    NUM_EXAMPLES = 3
    n_use = 30
    batch_size = 1
    num_steps = 32
    state_size = n_use
    num_epochs = 4
    test_frac = 0.2
    train_input = tf.placeholder(tf.float32, [batch_size,num_steps,n_use], name='input_placeholder')
    train_target = tf.placeholder(tf.float32, [batch_size,num_steps,n_use], name='labels_placeholder')

    # load data
    mat_file = sio.loadmat('data/10_timeseries.mat')
    stim_x = mat_file['stim']
    raw_x = mat_file['timeseries']
    #import pdb; pdb.set_trace()
    tot_neurons = np.shape(raw_x)[0]
    #import pdb; pdb.set_trace()
    raw_x = raw_x[-n_use:].reshape(n_use,-1,10).sum(axis=2)[:,:-10]
    mean_raw_x = np.expand_dims(raw_x.mean(axis=1),axis=1)
    mean_raw_x = np.repeat(mean_raw_x, num_steps, axis=1).reshape([num_steps,n_use])
    #stim_x = stim_x.reshape(1,-1,10).max(axis=2)[:,:-10]
    #raw_x = np.concatenate([raw_x, stim_x], axis=0)
    raw_y = np.roll(raw_x, -1, axis=1)
    epoch_size = np.size(raw_x, axis=-1) // num_steps
    train_idxs = np.random.randint(epoch_size, size=int((1-test_frac)*epoch_size))
    test_idxs = np.random.randint(epoch_size, size=int(test_frac*epoch_size))
    raw_data = (raw_x, raw_y)

    #train_input = inputs[:NUM_EXAMPLES, :, :-16]
    #train_output = targets[:NUM_EXAMPLES, :, :-16]

    #test_input = inputs[NUM_EXAMPLES:]
    #test_output = targets[NUM_EXAMPLES:]

    #train_output = np.transpose(train_output, [0,2,1])
    #train_input = np.transpose(train_input, [2,1,0])
    #train_input = np.expand_dims(train_input, axis=2)

    with tf.name_scope('Train'):
        with tf.variable_scope('Model', reuse=None):
            t = time.time()
            model = Model(train_input, train_target, mean_raw_x, num_steps,state_size)
            print("it took", time.time() - t, "seconds to build the Train graph")
    with tf.name_scope('Test'):
        with tf.variable_scope('Model', reuse=True):
            t = time.time()
            m_test = Model(train_input, train_target, mean_raw_x, num_steps, state_size)
            print("it took", time.time() - t, "seconds to build the Test graph")

    with tf.Session() as sess:
        t = time.strftime("%Y%m%d.%H.%M.%S",time.localtime(time.time()))
        train_writer = tf.summary.FileWriter('log/data_10/n30/'+t, sess.graph)
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx,epoch in enumerate(gen_epochs(raw_data,num_epochs,n_use,num_steps,train_idxs)):
            training_loss = 0
            print("EPOCH: %d" % idx)
            for step,(X,Y) in tqdm(enumerate(epoch)):
                #import pdb; pdb.set_trace()
                x_reshaped = np.expand_dims(X, axis=2)
                new_x = np.reshape(x_reshaped, [batch_size,num_steps,n_use])
                new_y = np.reshape(Y, [batch_size, num_steps, n_use])
                feed_dict = {train_input:new_x,
                             train_target:new_y}
                fetchers = {
                    #'total_loss': model.total_loss,
                    'prediction': model.prediction,
                    'summary': model.merge_summaries,
                    'debug': model.debug,
                    'eval':model.optimize
                }

                vals = model.do(sess,fetchers,feed_dict)
                rnn_inputs = vals["debug"]
                #import pdb; pdb.set_trace()
                #training_loss += vals['total_loss']
                if step % 1000 == 0 and step > 0:
                    #print("Epoch: %d Example: %d Error: %.3f" % (idx+1,step+1, 100*vals['total_loss']))
                    #print("Average loss at step %d for the last 250 steps: %.3f" % (step, training_loss/10))
                    if step % 1000 == 0:
                        print('\ntesting...')
                        for test_step, (X,Y) in tqdm(enumerate(gen_batch(raw_data, batch_size,num_steps, test_idxs))):
                            x_reshaped = np.expand_dims(X, axis=2)
                            new_x = np.reshape(x_reshaped, [batch_size,num_steps,n_use])
                            new_y = np.reshape(Y, [batch_size, num_steps, n_use])
                            test_feed_dict = {train_input:new_x,
                                         train_target:new_y}
                            test_fetchers = {
                                    'total_loss': m_test._total_loss,
                                    'fev': m_test.FEV
                                    }
                            test_vals = m_test.do(sess,test_fetchers, test_feed_dict)
                            if test_step % 500 == 0:
                                print('\nEPOCH: %d Test_Total_Loss: %.3f Test_FEV: %.3f' % (idx,test_vals['total_loss'],test_vals['fev']))
                        print('\ntesting...done')
                        print('\n','EPOCH: %d Total')
                    #training_losses.append(training_loss/10)
                    global_step = model.step(session=sess)
                    train_writer.add_summary(vals['summary'],global_step)
                    training_loss = 0

if __name__ == '__main__':
    main()
