import scipy.io as sio
import os
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import time
from tqdm import tqdm

from rnn_model import RecurrentActivityModel
from helpers.dataloaders import MatLoader as ld
from helpers.utils import *

def run(params,sess,num_epochs=1):
    data=params['data']
    idxs=params['idxs']
    batch_size=params['batch_size']
    num_steps=params['num_steps']
    n_use=params['neurons']
    train_input=params['train_input']
    train_target=params['train_target']
    model=params['model']
    train_writer=params['train_writer']
    for idx,epoch in enumerate(gen_epochs(num_epochs,data,idxs,batch_size,num_steps,n_use)):
        print("EPOCH: %d" % idx)
        for step,(X,Y) in tqdm(enumerate(epoch)):

            feed_dict = {train_input:X,
                         train_target:Y}
            fetchers = {
                'summary': model.merge_summaries,
                'eval':model.optimize
            }
            vals = model.do(sess,fetchers,feed_dict)
            global_step = model.step(session=sess)
            train_writer.add_summary(vals['summary'],global_step)


def main():

    # Set params/placeholders
    time_shuffle=True
    tf.reset_default_graph()
    NUM_EXAMPLES = 3
    n_use = 60
    batch_size = 10
    num_steps = 32
    state_size = n_use
    num_epochs = 20
    test_frac = 0.2
    train_input = tf.placeholder(tf.float32, [batch_size,num_steps,n_use], name='input_placeholder')
    train_target = tf.placeholder(tf.float32, [batch_size,n_use], name='labels_placeholder')

    # load data
    FILE = 'data/10_timeseries_trial_shuffled.mat'
    print('loading file: '+FILE)
    mat_file = sio.loadmat(FILE)
    raw_x = mat_file['timeseries']
    #import pdb; pdb.set_trace()
    tot_neurons = np.shape(raw_x)[0]
    #import pdb; pdb.set_trace()
    raw_x = raw_x[-n_use:].reshape(n_use,-1,10).sum(axis=2)[:,:-10]
    mean_raw_x = np.expand_dims(raw_x.mean(axis=1),axis=1)
    mean_raw_x = np.repeat(mean_raw_x, batch_size, axis=1).reshape([batch_size,n_use])
    #stim_x = stim_x.reshape(1,-1,10).max(axis=2)[:,:-10]
    #raw_x = np.concatenate([raw_x, stim_x], axis=0)
    raw_x_ts = []
    if time_shuffle==True:
        for n in np.arange(n_use):
            permuted = np.random.permutation(raw_x[n,:])
            raw_x_ts.extend([permuted])
        raw_x = np.array(raw_x_ts)
    raw_y = np.roll(raw_x, -1, axis=1)
    epoch_size = np.size(raw_x, axis=-1) // num_steps
    idxs = np.random.permutation(epoch_size)
    train_idxs = idxs[:int((1-test_frac)*epoch_size)]
    test_idxs = idxs[-int(test_frac*epoch_size):]
    raw_data = (raw_x, raw_y)

    with tf.name_scope('Train'):
        with tf.variable_scope('Model', reuse=None):
            t = time.time()
            model = RecurrentActivityModel(
                    train_input,
                    train_target,
                    mean_raw_x,
                    num_steps,
                    state_size,
                    learning_rate=1e-4)
            print("it took", time.time() - t, "seconds to build the Train graph")
    with tf.name_scope('Test'):
        with tf.variable_scope('Model', reuse=True):
            t = time.time()
            m_test = RecurrentActivityModel(train_input, train_target, mean_raw_x, num_steps, state_size)
            print("it took", time.time() - t, "seconds to build the Test graph")

    with tf.Session() as sess:
        t = time.strftime("%Y%m%d.%H.%M.%S",time.localtime(time.time()))
        TFR_PATH = 'log/data_10/n'+str(n_use)+'/'+t
        print('Logging to...',TFR_PATH)
        train_writer = tf.summary.FileWriter(TFR_PATH, sess.graph)
        sess.run(tf.global_variables_initializer())
        train_run_params = {
                'data':raw_data,
                'idxs':train_idxs,
                'batch_size':batch_size,
                'num_steps':num_steps,
                'train_target':train_target,
                'train_input':train_input,
                'model':model,
                'train_writer':train_writer,
                'neurons':n_use
                }
        test_run_params = dict(
                )
        out = []
        for i in np.arange(num_epochs):
            run(train_run_params,sess)

            print('\ntesting...')
            for epoch in gen_epochs(1,raw_data, test_idxs,batch_size,num_steps,n_use):
                for test_step, (X,Y) in enumerate(epoch):
                    test_feed_dict = {train_input:X,
                                     train_target:Y}
                    test_fetchers = {
                            'total_loss': m_test._total_loss,
                            'prediction': m_test._prediction,
                            'fev': m_test.FEV,
                            'summary': m_test.merge_summaries
                            }
                    test_vals = m_test.do(sess,test_fetchers, test_feed_dict)
                    global_step = model.step(session=sess)
                    train_writer.add_summary(test_vals['summary'],global_step)
                    if False:
                        print('adding to data to...',Fn)
                        for n in np.arange(n_use):
                            for i,(a,b,c) in enumerate(zip(X[:,:,n],Y[:,:,n],test_vals['prediction'][:,n])):
                                entry = {
                                        field_names[0]:global_step,
                                        field_names[1]:n,
                                        field_names[5]:i,
                                        field_names[2]:a,
                                        field_names[3]:b,
                                        field_names[4]:c
                                        }
                                out.extend([entry])
                        w.writerows(out)

            print('\ntesting...done')

if __name__ == '__main__':
    main()
