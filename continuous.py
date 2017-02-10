import scipy.io as sio
import os
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import time
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
from tqdm import tqdm

from rnn_model import RecurrentActivityModel
from helpers.dataloaders import MatLoader as ld
from helpers.utils import *

def run(params,sess,num_epochs=1):

    return vals



def main():

    # Set params/placeholders
    TIME_SHUFFLE=True
    VALIDATE=True
    STREAMING=False
    VERBOSE=False
    FLAGS=dict(
            TIME_SHUFFLE=TIME_SHUFFLE,
            VALIDATE=VALIDATE,
            STREAMING=STREAMING,
            VERBOSE=VERBOSE
            )

    tf.reset_default_graph()

    n_use = 30
    batch_size = 10
    num_steps = 39
    state_size = n_use
    num_epochs = 50
    next_n = 3
    binning=10
    test_frac = 0.2
    train_input = tf.placeholder(tf.float32, [batch_size,num_steps,n_use], name='input_placeholder')
    train_target= tf.placeholder(tf.float32, [batch_size,next_n,n_use], name='labels_placeholder')
    #state_tuple=

    # load data
    FILE = 'data/02_timeseries.mat'
    #FILE = 'data/10_timeseries.mat'
    #FILE = 'data/10_timeseries_trial_shuffled.mat'
    print('loading file: '+FILE)
    if VERBOSE:
        print('next_n=',next_n)
    mat_file = sio.loadmat(FILE)
    raw_x = mat_file['timeseries']
    tot_neurons = np.shape(raw_x)[0]
    raw_x = raw_x[:n_use].reshape(n_use,-1,binning).sum(axis=2)[:,:-binning]
    mean_raw_x = np.expand_dims(raw_x.mean(axis=1),axis=1)
    mean_raw_x = np.repeat(mean_raw_x, batch_size*next_n, axis=1).reshape([batch_size,next_n,n_use])
    #stim_x = stim_x.reshape(1,-1,10).max(axis=2)[:,:-10]
    #raw_x = np.concatenate([raw_x, stim_x], axis=0)
    raw_x_ts = []
    if TIME_SHUFFLE:
        for n in np.arange(n_use):
            permuted = np.random.permutation(raw_x[n,:])
            raw_x_ts.extend([permuted])
        raw_x = np.array(raw_x_ts)
    epoch_size = np.size(raw_x, axis=-1) // num_steps
    idxs = np.random.permutation(epoch_size)[:-(num_steps+next_n)]
    train_idxs = idxs[:int((1-test_frac)*epoch_size)]
    train_epoch_length = len(train_idxs)//batch_size
    test_idxs = idxs[-int(test_frac*epoch_size):]
    test_epoch_length = len(test_idxs)//batch_size
    raw_data = (raw_x, raw_x)

    if STREAMING:
        # setup streaming heatmap object
        #weight_matrix = np.zeros((n_use,n_use))
        FEV_2D = np.zeros((n_use,next_n))
        predictions = np.zeros((n_use,num_steps+next_n))
        input_data = np.zeros((n_use,num_steps+next_n))

        stream_ids = tls.get_credentials_file()['stream_ids']
        sid = stream_ids[0]
        sid1 = stream_ids[1]
        x_sid = stream_ids[2]
        predictions_stream = dict(token=sid1,maxpoints=n_use)
        hm_stream = dict(token=sid,maxpoints=n_use)
        x_input_stream = dict(token=x_sid,maxpoints=n_use)
        x_trace = go.Heatmap(
                z=input_data,
                y=np.arange(-num_steps,next_n)+1,
                zmax=3,
                zmin=0,
                colorscale='Jet',
                stream=x_input_stream
                )
        hm_trace = go.Heatmap(
                z=FEV_2D,
                colorscale='Viridis',
                stream=hm_stream
                )
        pred_trace = go.Heatmap(
                z=predictions,
                y=np.arange(-num_steps,next_n)+1,
                stream=predictions_stream,
                colorscale='Jet',
                zmax=3,
                zmin=0
                )
        #x_input_trace = go.
        data = go.Data([hm_trace])
        data1 = go.Data([pred_trace])
        x_data = go.Data([x_trace])

        fig = dict(
                data=data,

                layout=dict(
                title='2D FEV',
                yaxis=dict(
                    title='time'
                        ),
                xaxis=dict(
                    title='neurons'
                        )
                    )
                )
        pred_fig = dict(
                data=data1,

                layout=dict(
                    title='Predictions',

                    yaxis=dict(
                        title='time (%dms)' % (binning),
                        range=[-num_steps,next_n]
                        ),

                    xaxis=dict(
                        title='Neuron'
                        )
                    )
                )
        input_fig = dict(
                data=x_data,

                layout=dict(
                    title='True Values',

                    yaxis=dict(
                        title='time (%dms)' % (binning),
                        range=[-num_steps,next_n]
                        ),

                    xaxis=dict(
                        title='Neuron'
                        )
                    )
                )
        py.plot(fig, filename='2d-fev', auto_open=False)
        py.plot(pred_fig, filename='last_prediction', auto_open=False)
        py.plot(input_fig, filename='last-Y-streaming', auto_open=False)
        s = py.Stream(sid)
        pred_s = py.Stream(sid1)
        input_s = py.Stream(x_sid)
        streams = [s,pred_s,input_s]

    with tf.name_scope('Train'):
        with tf.variable_scope('Model', reuse=None):
            t = time.time()
            model = RecurrentActivityModel(
                    train_input,
                    train_target,
                    mean_raw_x,
                    num_steps,
                    state_size,
                    next_n,
                    learning_rate=.01)
            if VERBOSE:
                print("it took", time.time() - t, "seconds to build the Train graph")

    with tf.name_scope('Test'):
        with tf.variable_scope('Model', reuse=True):
            t = time.time()
            m_test = RecurrentActivityModel(train_input, train_target, mean_raw_x, num_steps, state_size,next_n)
            if VERBOSE:
                print("it took", time.time() - t, "seconds to build the Test graph")

    with tf.Session() as sess:
        t = time.strftime("%Y%m%d.%H.%M.%S",time.localtime(time.time()))
        TFR_PATH = 'log/data_02/n'+str(n_use)+'/'+t
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
                'next_n':next_n,
                'neurons':n_use
                }

        last_vals={'status':{'lr':1e-2}}
        for idx,epoch in enumerate(gen_epochs(num_epochs,raw_data,train_idxs,batch_size,num_steps,n_use,next_n,opts=FLAGS)):
            b_id = np.random.randint(batch_size)
            status = "EPOCH: %d LR: %.5f" % (idx,last_vals['status']['lr'])
            for step,(X,Y,XY) in tqdm(enumerate(epoch),desc=status,total=train_epoch_length):
                feed_dict = {train_input:X,
                             train_target:Y}
                fetchers = {
                    'summary':      model.merge_summaries,
                    'status':       model.status,
                    'fev_2d':       model.FEV_2d,
                    #'weights':      model._weight_matrix,
                    'prediction':   model.prediction,
                    'eval':         model.optimize
                }
                vals = model.do(sess,fetchers,feed_dict)
                last_vals = vals
                global_step = model.step(session=sess)
                train_writer.add_summary(vals['summary'],global_step)

                if step % 5000 == 0:
                    if STREAMING:
                        #r_x = np.reshape(X[b_id],[n_use,-1])
                        #r_p = np.reshape(vals['prediction'][b_id], [n_use,-1])
                        r_p = vals['prediction'][b_id,:,:]
                        prediction_ex=np.concatenate([X[b_id,:,:],r_p],axis=0)
                        true_ex=XY[b_id,:,:]

                        updates = [
                                dict(
                                    z=vals['fev_2d'],
                                    type='heatmap'),
                                dict(
                                    z=prediction_ex,
                                    type='heatmap'),
                                dict(
                                    z=true_ex,
                                    type='heatmap' )
                                ]
                        for stream,update in zip(streams,updates):
                            time.sleep(.0001)
                            stream.open()
                            stream.write(update)
                            time.sleep(.0001)
                            stream.close()

            if VALIDATE:
                # testing
                test_status = "EPOCH: %d testing..." % idx
                for epoch in gen_epochs(1,raw_data, test_idxs,batch_size,num_steps,n_use,next_n,opts=FLAGS):
                    for test_step, (X,Y,XY) in tqdm(enumerate(epoch),desc=test_status,total=test_epoch_length):
                        test_feed_dict = {train_input:X,
                                         train_target:Y}
                        test_fetchers = {
                                'total_loss': m_test._total_loss,
                                #'prediction': m_test._prediction,
                                #'weights': m_test._weight_matrix,
                                'fev': m_test.FEV,
                                'summary': m_test.merge_summaries
                                }
                        test_vals = m_test.do(sess,test_fetchers, test_feed_dict)
                        train_writer.add_summary(test_vals['summary'],global_step)

if __name__ == '__main__':
    main()
