# -*- coding: utf-8 -*-

import scipy.io as sio
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import time


class Model:

    def __init__(self, x, y, num_steps=50, state_size=25, learning_rate=1e-4, num_layers=3):
        # Config Variables
        self.x = x
        self.y = y
        self.num_steps = num_steps  # number of truncated backprop steps ('n')
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.num_classes = 2  # num of possible outputs (0,1) in this case.
        batch_size = self.x.get_shape()[0]


        
        embeddings = tf.get_variable('embedding_matrix',
                                    [self.num_classes, self.state_size])
        
        rnn_inputs = [tf.squeeze(i) for i in tf.split(1, num_steps, tf.nn.embedding_lookup(embeddings,x))]
        import pdb; pdb.set_trace()
        weight = tf.get_variable('weight',[self.state_size, self.num_classes])
        bias = tf.get_variable('bias', [self.num_classes], initializer=tf.constant_initializer(0.0))

        cell = tf.nn.rnn_cell.LSTMCell(self.state_size)
        #cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers,state_is_tuple=True)
        self.init_state = cell.zero_state(batch_size, tf.float32)
        #import pdb; pdb.set_trace()
        # rnn_outputs shape = (batch_size, num_steps, state_size)
        # final_state = rnn_outputs[:,-1,:] = (30, 25)
        rnn_outputs, final_state = tf.nn.rnn(
            cell,
            rnn_inputs,
            initial_state=self.init_state,
            dtype=tf.float32
        )
        self._debug = rnn_outputs.get_shape()
        # Flatten rnn_outputs down to shape = (batch_size*num_steps, state_size)
        #self._p_logits = tf.Print([self.x], [self.x], message="logits and logits_o")
        rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
        
        # Flatten y; (30,50) -> (30*50) = (1500)
        y_reshaped = tf.reshape(self.y, [-1])

        # (1500,25) x (25,2) = (1500,2)
        logits = tf.matmul(rnn_outputs, weight) + bias
        logits_o = tf.reshape(logits, [-1, self.num_steps, self.num_classes])
        #import pdb; pdb.set_trace()
        seqw =  tf.ones((batch_size, num_steps))
        self._prediction = tf.nn.softmax(logits_o)
        #self._prediction = tf.reshape(self._prediction, [-1, num_classes])
        with tf.name_scope('cross_entropy'):
            self._error = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_reshaped)
            with tf.name_scope('total'):
                self._total_loss = tf.reduce_mean(self._error)
        seq_loss = tf.nn.seq2seq.sequence_loss_by_example(
            tf.unpack(logits_o, axis=1),
            tf.unpack(self.y, axis=1), 
            tf.unpack(seqw, axis=1),
            average_across_timesteps=False
        )
        #  import pdb; pdb.set_trace()
        perplexity = tf.exp(seq_loss)
        #import pdb; pdb.set_trace()
        self._avg_perplexity = tf.reduce_mean(perplexity)
        tf.summary.scalar('total_loss', self._total_loss)
        tf.summary.scalar('avg_perplexity', self._avg_perplexity)
        self._optimize = tf.train.AdamOptimizer(learning_rate).minimize(self._avg_perplexity, global_step=global_step)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(self.prediction,2), tf.cast(self.y, dtype=tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

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

    @property
    def debug(self):
        return self._debug

def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = np.size(raw_x,axis=-1)
    #raw_x = np.squeeze(raw_x)
    #raw_y = np.squeeze(raw_y)

    # partition raw data into batches and stack them vertically in a data matrix
    #batch_partition_length = data_length // num_steps
    #data_x = np.zeros([num_steps, batch_size, batch_partition_length], dtype=np.int32)
    #data_y = np.zeros([num_steps, batch_size, batch_partition_length], dtype=np.int32)
    #for i in range(batch_size):
    #    data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
    #    data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    #import pdb; pdb.set_trace()
    # further divide batch partitions into num_steps for truncated backprop
    data_x, data_y = raw_x, raw_y
    epoch_size = data_length // num_steps
    #import pdb; pdb.set_trace()
    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        #import pdb;pdb.set_trace()
        #import pdb; pdb.set_trace()
        yield (x, y)

def gen_epochs(raw_data,n, batch_size,num_steps):
    for i in range(n):
        yield gen_batch(raw_data, batch_size, num_steps)

def main():
    #%%
    # load data
    tf.reset_default_graph()
    NUM_EXAMPLES = 3
    batch_size = 3
    state_size = 8
    num_steps = 96
    num_epochs = 4
    train_input = tf.placeholder(tf.int32, [batch_size,num_steps], name='input_placeholder')
    train_target = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    mat_file = sio.loadmat('data/10_timeseries.mat')

    raw_x = mat_file['timeseries'][:,:-10]
    raw_y = np.roll(mat_file['timeseries'], -1, axis=1)[:,:-10]
    raw_data = (raw_x[:batch_size], raw_y[:batch_size])
    #import pdb; pdb.set_trace() 
    #train_input = inputs[:NUM_EXAMPLES, :, :-16]
    #train_output = targets[:NUM_EXAMPLES, :, :-16]

    #test_input = inputs[NUM_EXAMPLES:]
    #test_output = targets[NUM_EXAMPLES:]
    # Set params/placeholders

    #train_output = np.transpose(train_output, [0,2,1])
    #train_input = np.transpose(train_input, [2,1,0])
    #train_input = np.expand_dims(train_input, axis=2)

    with tf.name_scope('Train'):
        with tf.variable_scope('Model', reuse=None):
            t = time.time()
            model = Model(train_input, train_target, num_steps,state_size)
            print("it took", time.time() - t, "seconds to build the graph")

    with tf.Session() as sess:
        t = time.strftime("%Y%m%d.%H.%M.%S",time.localtime(time.time()))
        train_writer = tf.summary.FileWriter('log/'+t, sess.graph)
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx,epoch in enumerate(gen_epochs(raw_data,num_epochs,batch_size,num_steps)):
            training_loss = 0
            print("EPOCH: %d" % idx)
            for step,(X,Y) in enumerate(epoch):
                #import pdb; pdb.set_trace()
                feed_dict = {train_input:X,
                             train_target:Y}
                fetchers = {
                'total_loss': model.total_loss,
                'prediction': model.prediction,
                'summary': model.merge_summaries,
                'debug': model.debug, 
                'eval':model.optimize
                }
                #import pdb; pdb.set_trace()
                vals = model.do(sess,fetchers,feed_dict)

                training_loss += vals['total_loss']
                if step % 10 == 0 and step > 0:
                    #print("Epoch: %d Example: %d Error: %.3f" % (idx+1,step+1, 100*vals['total_loss']))
                    #print("Average loss at step %d for the last 250 steps: %.3f" % (step, training_loss/10))
                    print(vals['debug'])
                    #training_losses.append(training_loss/10)
                    global_step = model.step(session=sess)
                    train_writer.add_summary(vals['summary'],global_step)
                    training_loss = 0

if __name__ == '__main__':
    main()
