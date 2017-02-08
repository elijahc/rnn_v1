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
    data=params['data']
    idxs=params['idxs']
    batch_size=params['batch_size']
    num_steps=params['num_steps']
    n_use=params['neurons']
    train_input=params['train_input']
    train_target=params['train_target']
    model=params['model']
    train_writer=params['train_writer']
    X,Y = params['step']
    next_n=params['next_n']

    feed_dict = {train_input:X,
                 train_target:Y}
    fetchers = {
        'summary': model.merge_summaries,
        'weights': model._weight_matrix,
        'prediction': model._prediction,
        'eval':model.optimize
    }
    vals = model.do(sess,fetchers,feed_dict)
    global_step = model.step(session=sess)
    train_writer.add_summary(vals['summary'],global_step)

    return vals



def main():

    # Set params/placeholders
    time_shuffle=False
    validate=False

    tf.reset_default_graph()

    n_use = 60
    batch_size = 1
    num_steps = 30
    state_size = n_use
    num_epochs = 100
    next_n = 5
    binning=10
    test_frac = 0.2
    train_input = tf.placeholder(tf.float32, [batch_size,num_steps,n_use], name='input_placeholder')
    train_target = tf.placeholder(tf.float32, [batch_size,next_n,n_use], name='labels_placeholder')

    # load data
    #FILE = 'data/02_timeseries.mat'
    FILE = 'data/10_timeseries.mat'
    #FILE = 'data/10_timeseries_trial_shuffled.mat'
    print('loading file: '+FILE)
    print('next_n=',next_n)
    mat_file = sio.loadmat(FILE)
    raw_x = mat_file['timeseries']
    #import pdb; pdb.set_trace()
    tot_neurons = np.shape(raw_x)[0]
    #import pdb; pdb.set_trace()
    raw_x = raw_x[-n_use:].reshape(n_use,-1,binning).sum(axis=2)[:,:-binning]
    mean_raw_x = np.expand_dims(raw_x.mean(axis=1),axis=1)
    mean_raw_x = np.repeat(mean_raw_x, batch_size*next_n, axis=1).reshape([batch_size,next_n,n_use])
    #stim_x = stim_x.reshape(1,-1,10).max(axis=2)[:,:-10]
    #raw_x = np.concatenate([raw_x, stim_x], axis=0)
    raw_x_ts = []
    if time_shuffle==True:
        for n in np.arange(n_use):
            permuted = np.random.permutation(raw_x[n,:])
            raw_x_ts.extend([permuted])
        raw_x = np.array(raw_x_ts)
    epoch_size = np.size(raw_x, axis=-1) // num_steps
    idxs = np.random.permutation(epoch_size)[:-num_steps+next_n]
    train_idxs = idxs[:int((1-test_frac)*epoch_size)]
    test_idxs = idxs[-int(test_frac*epoch_size):]
    raw_data = (raw_x, raw_x)

    # setup streaming heatmap object
    weight_matrix = np.zeros((n_use,n_use))
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
            zmax=3,
            zmin=0,
            colorscale='Jet',
            stream=x_input_stream
            )
    hm_trace = go.Heatmap(
            z=weight_matrix,
            colorscale='Viridis',
            zmax=2,
            zmin=-1,
            stream=hm_stream
            )
    pred_trace = go.Heatmap(
            z=predictions,
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
            title='Last Weight Matrix',
            yaxis=dict(
                title='states'
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
                    title='neuron')
            ))
    input_fig = dict(
            data=x_data,

            layout=dict(
                title='True Values',
                yaxis=dict(
                    title='neuron'
                    )
                )
            )
    py.plot(fig, filename='last-weight-matrix-streaming', auto_open=False)
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
                    learning_rate=1e-4)
            print("it took", time.time() - t, "seconds to build the Train graph")
    with tf.name_scope('Test'):
        with tf.variable_scope('Model', reuse=True):
            t = time.time()
            m_test = RecurrentActivityModel(train_input, train_target, mean_raw_x, num_steps, state_size,next_n)
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
        test_run_params = dict(
                )
        out = []
        for idx,epoch in enumerate(gen_epochs(num_epochs,raw_data,idxs,batch_size,num_steps,n_use,next_n)):
            print("EPOCH: %d training..." % idx)
            for step,(X,Y,XY) in tqdm(enumerate(epoch)):
                train_run_params['step'] = (X,Y)
                vals=run(train_run_params,sess)

                if step % 1000 == 0:
                    b_id = np.random.randint(batch_size)
                    r_x = np.reshape(X[b_id],[n_use,-1])
                    r_p = np.reshape(vals['prediction'][b_id], [n_use,-1])
                    prediction_ex=np.concatenate([r_x,r_p],axis=1)
                    true_ex=np.reshape(XY[b_id],[n_use,-1])
                    updates = [
                            dict(
                                z=vals['weights'],
                                type='heatmap'),
                            dict(
                                z=np.squeeze(prediction_ex),
                                type='heatmap'),
                            dict(
                                z=np.squeeze(true_ex),
                                type='heatmap' )
                            ]
                    for stream,update in zip(streams,updates):
                        time.sleep(.0001)
                        stream.open()
                        stream.write(update)
                        time.sleep(.0001)
                        stream.close()

            if validate==True:
                print("EPOCH: %d testing..." % idx)
                # testing
                for epoch in gen_epochs(1,raw_data, test_idxs,batch_size,num_steps,n_use,next_n):
                    for test_step, (X,Y) in enumerate(epoch):
                        test_feed_dict = {train_input:X,
                                         train_target:Y}
                        test_fetchers = {
                                'total_loss': m_test._total_loss,
                                'prediction': m_test._prediction,
                                'weights': m_test._weight_matrix,
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

if __name__ == '__main__':
    main()
