import scipy.io as sio
import pprint
from scipy.optimize import brute
import os
import json
import pickle
import configparser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from tqdm import tqdm

from rnn_model import RecurrentActivityModel
from helpers.dataloaders import MatLoader as ld
from helpers.utils import *
from sklearn.preprocessing import MinMaxScaler

import argparse
import time

def main():
    FLAGS = init_configs()
    if FLAGS.OPTIMIZE:
        max_s = 15
        batch_span = slice(2,max_s,1)
        epochs_span = slice(1,10,2)
        #state_span  = (1,10)
        #layers_span = (1,10)
        a,f,g,j = brute(optimize,(batch_span,epochs_span),args=tuple([FLAGS]),full_output=True)
        full_out = {'x0':a,'fval':f,'grid':g,'Jout':j}
        pprint.pprint(full_out)
        full_out
    else:
        train(FLAGS)


def optimize(x,FLAGS):
    FLAGS.batch_size = int(x[0])
    FLAGS.epochs = int(x[1])
    #FLAGS.state_size = x[2]
    #FLAGS.layers_span = x[3]

    print('starting run with:')
    pprint.pprint(vars(FLAGS))

    tf = train(FLAGS)
    return np.array(tf).mean()

def train(FLAGS):

    # Set params/placeholders
    FLAGS.rnn_size = FLAGS.state_size
    FLAGS.batch = FLAGS.batch_size
    FLAGS.seq_len = FLAGS.num_steps
    FLAGS.next_n = FLAGS.guess
    TIME_SHUFFLE=FLAGS.TIME_SHUFFLE
    VALIDATE=FLAGS.VALIDATE
    STREAMING=FLAGS.STREAMING
    VERBOSE=FLAGS.VERBOSE

    tf.reset_default_graph()

    n_use = FLAGS.n_use
    if FLAGS.STIM_CUE:
        an_use = n_use-1
    else:
        an_use = n_use
    batch_size = FLAGS.batch_size
    num_steps = FLAGS.seq_len
    state_size = FLAGS.rnn_size
    num_epochs = FLAGS.epochs
    next_n = FLAGS.guess
    binning = FLAGS.bin_size
    test_frac = FLAGS.test
    learning_rate = FLAGS.lr
    train_input = tf.placeholder(tf.float32, [batch_size,num_steps,n_use], name='input_placeholder')
    train_target= tf.placeholder(tf.float32, [batch_size,next_n,n_use], name='labels_placeholder')
    #state_tuple=

    # load data
    FILE = FLAGS.infile
    #FILE = 'data/10_timeseries.mat'
    #FILE = 'data/10_timeseries_trial_shuffled.mat'
    print('loading file: '+FILE)
    if VERBOSE:
        print('next_n=',next_n)
    mat_file = sio.loadmat(FILE)

    scaler = MinMaxScaler(feature_range=(0,1))

    raw_x = mat_file['timeseries']
    stim_x = mat_file['stim']

    tot_neurons = np.shape(raw_x)[0]
    stim_x = stim_x.reshape(1,-1,binning).mean(axis=2)[:,:-binning]
    raw_x = raw_x[-an_use:].reshape(an_use,-1,binning).sum(axis=2)[:,:-binning]
    if FLAGS.STIM_CUE:
        raw_x = np.concatenate([raw_x,stim_x],axis=0)
    raw_x = scaler.fit_transform(raw_x)
    mean_raw_x = np.expand_dims(raw_x.mean(axis=1),axis=1)
    mean_raw_x = np.repeat(mean_raw_x, batch_size*next_n, axis=1).reshape([batch_size,next_n,n_use])
    #stim_x = stim_x.reshape(1,-1,10).max(axis=2)[:,:-10]
    #raw_x = np.concatenate([raw_x, stim_x], axis=0)
    raw_x_ts = []
    if TIME_SHUFFLE:
        for n in np.arange(n_use):
            permuted = np.random.permutation(raw_x[n,:])
            raw_x_ts.extend([permuted])
        raw_x_ts = np.array(raw_x_ts)
    epoch_size = np.size(raw_x, axis=-1) // num_steps
    idxs = np.random.permutation(epoch_size)[:-(num_steps+next_n)]
    train_idxs = idxs[:int((1-test_frac)*epoch_size)]
    train_epoch_length = len(train_idxs)//batch_size
    test_idxs = idxs[-int(test_frac*epoch_size):]
    test_epoch_length = len(test_idxs)//batch_size
    if TIME_SHUFFLE:
        raw_data = raw_x_ts
    else:
        raw_data = raw_x


    if STREAMING:
        # setup streaming heatmap object
        weight_matrix = np.zeros((n_use,n_use))
        #FEV_2D = np.zeros((n_use,next_n))
        predictions = np.zeros((n_use,num_steps+next_n))
        input_data = np.zeros((n_use,num_steps+next_n))

        #stream_ids = tls.get_credentials_file()['stream_ids']
        #sid = stream_ids[0]
        #sid1 = stream_ids[1]
        #x_sid = stream_ids[2]
        #predictions_stream = dict(token=sid1,maxpoints=n_use)
        #hm_stream = dict(token=sid,maxpoints=n_use)
        #x_input_stream = dict(token=x_sid,maxpoints=n_use)
        x_trace = go.Heatmap(
                z=input_data,
                y=np.arange(-num_steps,next_n)+1,
                zmax=3,
                zmin=0,
                colorscale='Magma',
                )
        hm_trace = go.Heatmap(
                z=weight_matrix,
                colorscale='Viridis',
                )
        pred_trace = go.Heatmap(
                z=predictions,
                y=np.arange(-num_steps,next_n)+1,
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
                title='weight_matrix',
                yaxis=dict(
                    title='state'
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
        #plot(fig, filename='last-weight-matrix-streaming')
        #plot(pred_fig, filename='last_prediction')
        #plot(input_fig, filename='last-Y-streaming')
        #s = py.Stream(sid)
        #pred_s = py.Stream(sid1)
        #input_s = py.Stream(x_sid)
        #streams = [s,pred_s,input_s]

    with tf.name_scope('Train'):
        with tf.variable_scope('Model', reuse=None):
            t = time.time()
            model = RecurrentActivityModel(
                    train_input,
                    train_target,
                    mean_raw_x,
                    FLAGS=FLAGS)
            if VERBOSE:
                print("it took", time.time() - t, "seconds to build the Train graph")

    with tf.name_scope('Test'):
        with tf.variable_scope('Model', reuse=True):
            t = time.time()
            m_test = RecurrentActivityModel(
                    train_input,
                    train_target,
                    mean_raw_x,
                    FLAGS=FLAGS)
            if VERBOSE:
                print("it took", time.time() - t, "seconds to build the Test graph")

    with tf.Session() as sess:
        t = time.strftime("%Y%m%d.%H.%M.%S",time.localtime(time.time()))
        SLUG_PATH = FLAGS.tag+'/data_' + str(FLAGS.data_set).zfill(2) + '/t_steps'+str(num_steps)+'/'+t
        TFR_PATH = 'log/'+SLUG_PATH
        print('Logging to...',TFR_PATH)
        train_writer = tf.summary.FileWriter(TFR_PATH, sess.graph)


        sess.run(tf.global_variables_initializer())

        last_vals={'status':{'lr':FLAGS.lr}}
        final_data = []
        test_fev = []
        for idx,epoch in enumerate(gen_epochs(num_epochs,raw_data,train_idxs,FLAGS)):
            epoch_output = []
            b_id = np.random.randint(batch_size)
            status = "EPOCH: %d LR: %.5f" % (idx,last_vals['status']['lr'])
            for step,(X,Y,XY) in tqdm(enumerate(epoch),desc=status,total=train_epoch_length):
                feed_dict = {train_input:X,
                             train_target:Y,
                             }
                fetchers = {
                    'summary':      model.merge_summaries,
                    'status':       model.status,
                    #'weights':      model._weight_matrix,
                    #'prediction':   model.prediction,
                    'eval':         model.optimize
                }
                vals = model.do(sess,fetchers,feed_dict)
                last_vals = vals
                global_step = model.step(session=sess)
                status=vals['status']
                #import pdb; pdb.set_trace()
                train_writer.add_summary(vals['summary'],global_step)
                if step % 100 == 0:
                    epoch_output.extend([ vals['status'] ])
                if global_step % 1000 == 0:

                    if STREAMING:
                        #r_x = np.reshape(X[b_id],[n_use,-1])
                        #r_p = np.reshape(vals['prediction'][b_id], [n_use,-1])
                        r_p = vals['prediction'][b_id,:,:]
                        prediction_ex=np.concatenate([X[b_id,:,:-1],r_p],axis=0)
                        true_ex=XY[b_id,:,:]

                        updates = [
                                dict(
                                    z=vals['weights'],
                                    type='heatmap'),
                                dict(
                                    z=prediction_ex,
                                    type='heatmap'),
                                dict(
                                    z=true_ex,
                                    type='heatmap' )
                                ]
                        #for stream,update in zip(streams,updates):
                        #    time.sleep(.0001)
                        #    stream.open()
                        #    stream.write(update)
                        #    time.sleep(.0001)
                        #    stream.close()
            if idx == 1:
                with open(TFR_PATH+'/config.json','w') as fp:
                    json.dump(vars(FLAGS),fp,sort_keys=True, indent=4)

            output_fp = TFR_PATH+'/bdump.p'
            if VERBOSE:
                print('saving '+output_fp+'...')
            final_data.extend([ epoch_output ])
            with open(output_fp,'wb') as fp:
                pickle.dump(np.array(final_data),fp)

            if VALIDATE:
                # testing
                test_status = "EPOCH: %d testing..." % idx
                for epoch in gen_epochs(1,raw_data, test_idxs,FLAGS):
                    for test_step, (X,Y,XY) in tqdm(enumerate(epoch),desc=test_status,total=test_epoch_length):
                        test_feed_dict = {train_input:X,
                                         train_target:Y}
                        test_fetchers = {
                                'total_loss': m_test._total_loss,
                                #'prediction': m_test._prediction,
                                #'weights': m_test._weight_matrix,
                                'total_loss': m_test._total_loss,
                                'summary': m_test.merge_summaries
                                }
                        test_vals = m_test.do(sess,test_fetchers, test_feed_dict)
                        test_fev.extend([test_vals['total_loss']])
                        train_writer.add_summary(test_vals['summary'],global_step)
    return test_fev

def init_configs():

    parser = argparse.ArgumentParser(description='RNN for modeling neuron populations')

    # Boolean arguments
    parser.add_argument('--verbose', dest='VERBOSE', action='store_true',
                        help='print all the things')
    parser.add_argument('--no-validate', dest='VALIDATE', action='store_false',
                        help='do not validate model after every epoch')
    parser.add_argument('--stream', dest='STREAMING', action='store_true',
                        help='stream results in realtime to plot.ly')
    parser.add_argument('--with-stim-cue', dest='STIM_CUE', action='store_true',
                        help='Include stim cue as 1 or 0 for each time step')
    parser.add_argument('--shuffle', dest='TIME_SHUFFLE', action='store_true',
                        help='time shuffle all the values of each neuron across the entire timeseries')
    parser.add_argument('--optimize', dest='OPTIMIZE', action='store_true',
                        help='Do a gridsearch of hardcoded params')
    parser.add_argument('--dev', dest='DEV', action='store_true',
                        help='Use the Dev set of config values for running')
    parser.set_defaults(VERBOSE=False,
                        STREAMING=False,
                        TIME_SHUFFLE=False,
                        STIM_CUE=False,
                        VALIDATE=True,
                        OPTIMIZE=False
                        )

    parser.add_argument('--tag', type=str, default='default',
                        help='tagline for the set of data analysis')
    parser.add_argument('--data_set', type=int, default=0,
                        help='numeric value (1-10) indicated which dataset')
    parser.add_argument('--state_size', type=int, default=30,
                        help='size of RNN hidden state')
    parser.add_argument('--n_use', type=int, default=30,
                        help='number of neurons to use')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='minibatch size')
    parser.add_argument('--num_steps', type=int, default=32,
                        help='RNN sequence length')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--cell', type=str, default='lstm', choices=['lstm','gru','bnlstm'],
                        help='Cell type for rnn')
    parser.add_argument('--guess', type=int, default=1,
                        help='number of sequences forward to guess')
    parser.add_argument('--test', type=float, default=0.2,
                        help='percentage of the dataset to set aside for testing')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial Learning Rate')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Num Layers in RNN')
    parser.add_argument('--bin_size', type=int, default=10,
                        help='Size of bin for resampling input data(1000Hz)')
    parser.add_argument('infile', metavar='infile', type=str,
                        help='Input data file path')

    FLAGS = parser.parse_args()
    return FLAGS

sys.excepthook = info

if __name__ == '__main__':
    main()
