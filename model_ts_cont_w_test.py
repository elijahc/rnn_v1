import rnn_model
import scipy.io as sio
from PIL import Image
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import time
from tqdm import tqdm

from helpers.utils import *

def main():

    # Set params/placeholders
    validate=False
    time_shuffle=False
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
    FILE = 'data/10_timeseries.mat'
    print('loading file: '+FILE)
    mat_file = sio.loadmat(FILE)
    #stim_x = mat_file['stim']
    raw_x = mat_file['timeseries']
    #import pdb; pdb.set_trace()
    tot_neurons = np.shape(raw_x)[0]
    #import pdb; pdb.set_trace()
    raw_x = raw_x[-n_use:].reshape(n_use,-1,10).sum(axis=2)[:,:-10]
    mean_raw_x = np.expand_dims(raw_x.mean(axis=1),axis=1)
    mean_raw_x = np.repeat(mean_raw_x, num_steps, axis=1).reshape([num_steps,n_use])
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
            model = RecurrentActivityModel(train_input, train_target, mean_raw_x, num_steps,state_size)
            print("it took", time.time() - t, "seconds to build the Train graph")
    with tf.name_scope('Test'):
        with tf.variable_scope('Model', reuse=True):
            t = time.time()
            m_test = RecurrentActivityModel(train_input, train_target, mean_raw_x, num_steps, state_size)
            print("it took", time.time() - t, "seconds to build the Test graph")

    # setup streaming heatmap object
    weight_matrix = np.zeros((30,30))
    predictions = np.zeros((32,30))
    input_data = np.zeros((32,30))
    stream_ids = tls.get_credentials_file()['stream_ids']
    sid = stream_ids[0]
    sid1 = stream_ids[1]
    x_sid = stream_ids[2]
    predictions_stream = dict(token=sid1,maxpoints=30)
    hm_stream = dict(token=sid,maxpoints=30)
    x_input_stream = dict(token=x_sid,maxpoints=30)
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
            zmax=5,
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
            title='Last Weight Matrix')
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
                title='input_data',
                yaxis=dict(
                    title='neuron'
                    )
                )
            )
    py.plot(fig, filename='last-weight-matrix-streaming', auto_open=False)
    py.plot(pred_fig, filename='last_prediction', auto_open=False)
    py.plot(input_fig, filename='last-input-streaming', auto_open=False)
    s = py.Stream(sid)
    pred_s = py.Stream(sid1)
    input_s = py.Stream(x_sid)
    streams = [s,pred_s,input_s]
    with tf.Session() as sess:
        t = time.strftime("%Y%m%d.%H.%M.%S",time.localtime(time.time()))
        TFR_PATH = 'log/data_10/n'+str(n_use)+'/'+t
        print('Logging to...',TFR_PATH)
        train_writer = tf.summary.FileWriter(TFR_PATH, sess.graph)
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx,epoch in enumerate(gen_epochs(raw_data,num_epochs,n_use,num_steps,train_idxs)):
            training_loss = 0
            print("EPOCH: %d" % idx)
            for step,(X,Y) in tqdm(enumerate(epoch)):
                x_reshaped = np.expand_dims(X, axis=2)
                new_x = np.reshape(x_reshaped, [batch_size,num_steps,n_use])
                new_y = np.reshape(Y, [batch_size, num_steps, n_use])
                feed_dict = {train_input:new_x,
                             train_target:new_y}
                fetchers = {
                    #'total_loss': model.total_loss,
                    'prediction': model.prediction,
                    'summary': model.merge_summaries,
                    'weights': model._weight_matrix,
                    #'debug': model.debug,
                    'eval':model.optimize
                }

                vals = model.do(sess,fetchers,feed_dict)
                #rnn_inputs = vals["debug"]
                #import pdb; pdb.set_trace()
                #training_loss += vals['total_loss']
                global_step = model.step(session=sess)
                weight_matrix = vals['weights']
                predictions = vals['prediction']
                if step % 5000 == 0 and step > 0:
                    updates = [
                            dict(
                                z=weight_matrix,
                                type='heatmap'),
                            dict(
                                z=predictions,
                                type='heatmap'),
                            dict(
                                z=np.squeeze(new_x),
                                type='heatmap' )
                            ]
                    for stream,update in zip(streams,updates):
                        time.sleep(.001)
                        stream.open()
                        stream.write(update)
                        stream.close()

                    #print("Epoch: %d Example: %d Error: %.3f" % (idx+1,step+1, 100*vals['total_loss']))
                    #print("Average loss at step %d for the last 250 steps: %.3f" % (step, training_loss/10))
                    if step % 1000 == 0 and validate==True:
                        print('\ntesting...')
                        for test_step, (X,Y) in enumerate(gen_batch(raw_data, batch_size,num_steps, test_idxs)):
                            x_reshaped = np.expand_dims(X, axis=2)
                            new_x = np.reshape(x_reshaped, [batch_size,num_steps,n_use])
                            new_y = np.reshape(Y, [batch_size, num_steps, n_use])
                            test_feed_dict = {train_input:new_x,
                                         train_target:new_y}
                            test_fetchers = {
                                    'total_loss': m_test._total_loss,
                                    'fev': m_test.FEV,
                                    'weight': m_test._weight_matrix,
                                    'summary': m_test.merge_summaries
                                    }
                            test_vals = m_test.do(sess,test_fetchers, test_feed_dict)
                            weight_matrix = test_vals['weight']
                            #if test_step % 500 == 0:
                                #print('\nEPOCH: %d Test_Total_Loss: %.3f Test_FEV: %.3f' % (idx,test_vals['total_loss'],test_vals['fev']))
                        print('\ntesting...done')
                        train_writer.add_summary(test_vals['summary'],global_step)
                train_writer.add_summary(vals['summary'],global_step)
        import pdb; pdb.set_trace()
        data = [go.Heatmap(z=weight_matrix)]
        url=py.plot(data,filename='last-weight-matrix')

if __name__ == '__main__':
    main()
