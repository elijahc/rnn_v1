import scipy.io as sio
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
from tqdm import tqdm

from embedded_rnn_model import RecurrentActivityModel
from helpers.dataloaders import MatLoader as ld
from helpers.utils import *

import argparse
import time

def main():
    parser = argparse.ArgumentParser(description='RNN for modeling neuron populations')

    # Boolean arguments
    parser.add_argument('--verbose', dest='VERBOSE', action='store_true',
                        help='print all the things')
    parser.add_argument('--no_validate', dest='VALIDATE', action='store_false',
                        help='do not validate model after every epoch')
    parser.add_argument('--stream', dest='STREAMING', action='store_true',
                        help='stream results in realtime to plot.ly')
    parser.add_argument('--shuffle', dest='TIME_SHUFFLE', action='store_true',
                        help='time shuffle all the values of each neuron across the entire timeseries')
    parser.set_defaults(VERBOSE=False,
                        STREAMING=False,
                        TIME_SHUFFLE=False,
                        VALIDATE=True
                        )

    parser.add_argument('--rnn_size', type=int, default=30,
                        help='size of RNN hidden state')
    parser.add_argument('--n_use', type=int, default=30,
                        help='number of neurons to use')
    parser.add_argument('--batch', type=int, default=10,
                        help='minibatch size')
    parser.add_argument('--seq_len', type=int, default=32,
                        help='RNN sequence length')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--guess', type=int, default=1,
                        help='number of sequences forward to guess')
    parser.add_argument('--test', type=float, default=0.2,
                        help='percentage of the dataset to set aside for testing')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial Learning Rate')
    parser.add_argument('--layers', type=float, default=1,
                        help='Num Layers in RNN')
    parser.add_argument('--bin_size', type=float, default=10,
                        help='Size of bin for resampling input data(1000Hz)')

    FLAGS = parser.parse_args()
    train(FLAGS)

def train(FLAGS):

    # Set params/placeholders
    TIME_SHUFFLE=FLAGS.TIME_SHUFFLE
    VALIDATE=FLAGS.VALIDATE
    STREAMING=FLAGS.STREAMING
    VERBOSE=FLAGS.VERBOSE

    tf.reset_default_graph()

    n_use = FLAGS.n_use
    batch_size = FLAGS.batch
    num_steps = FLAGS.seq_len
    state_size = FLAGS.rnn_size
    num_epochs = FLAGS.epochs
    next_n = FLAGS.guess
    binning = FLAGS.bin_size
    test_frac = FLAGS.test
    learning_rate = FLAGS.lr
    train_input = tf.placeholder(tf.int32, [batch_size,num_steps], name='input_placeholder')
    train_target= tf.placeholder(tf.int32, [batch_size,n_use], name='labels_placeholder')
    x_set_dim= tf.placeholder(tf.int32, shape=(), name='x_set_dim_placeholder')
    y_set_dim= tf.placeholder(tf.int32, shape=(), name='y_set_dim_placeholder')
    #state_tuple=

    # load data
    #FILE = 'data/02_timeseries.mat'
    FILE = 'data/10_timeseries.mat'
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
    raw_data = raw_x
    train_vec_set = vec_set(raw_x,FLAGS,idxs=train_idxs)
    test_vec_set = vec_set(raw_x,FLAGS,idxs=test_idxs)

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
                colorscale='Magma',
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
                colorscale='Magma',
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
                        routputange=[-num_steps,next_n]
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
                    len(train_vec_set['t_map']),
                    len(train_vec_set['n_map']),
                    FLAGS)
            if VERBOSE:
                print("it took", time.time() - t, "seconds to build the Train graph")

    with tf.name_scope('Test'):
        with tf.variable_scope('Model', reuse=True):
            t = time.time()
            m_test = RecurrentActivityModel(
                    train_input,
                    train_target,
                    mean_raw_x,
                    len(test_vec_set['t_map']),
                    len(test_vec_set['n_map']),
                    FLAGS)
            if VERBOSE:
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
                'next_n':next_n,
                'neurons':n_use
                }

        last_vals={'status':{'lr':1e-2}}
        for idx,epoch in enumerate(gen_epochs(num_epochs,raw_data,train_idxs,batch_size,num_steps,n_use,next_n,FLAGS)):
            b_id = np.random.randint(batch_size)
            status = "EPOCH: %d LR: %.5f" % (idx,last_vals['status']['lr'])
            for step,(X,Y,XY,lookups) in tqdm(enumerate(epoch),desc=status,total=train_epoch_length):
                FLAGS.x_set_dim = len(lookups['t_map'])
                FLAGS.y_set_dim = len(lookups['n_map'])
                import pdb; pdb.set_trace()
                feed_dict = {train_input:X,
                             train_target:Y,
                             x_set_dim:FLAGS.x_set_dim,
                             y_set_dim:FLAGS.y_set_dim
                             }
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
                for epoch in gen_epochs(1,raw_data, test_idxs,batch_size,num_steps,n_use,next_n,FLAGS):
                    for test_step, (X,Y,XY,lookups) in tqdm(enumerate(epoch),desc=test_status,total=test_epoch_length):
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
