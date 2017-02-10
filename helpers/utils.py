import numpy as np

def gen_batch(raw_data, idxs, batch_size, num_steps, n_use,next_n,VERBOSE=False):
    #raw_x = np.squeeze(raw_x)
    #raw_y = np.squeeze(raw_y)

    # partition raw data into batches and stack them vertically in a data matrix
    num_idxs = np.shape(idxs)[0]
    batch_partition_length = num_idxs // batch_size
    batch_idxs = np.random.permutation(num_idxs)

    for i in np.arange(batch_partition_length):
        xy = raw_data[i*batch_size:(i+1)*batch_size]
        x = xy[:,:num_steps,:]
        y = xy[:,-next_n:,:]
        yield (x,y,xy)


def gen_epochs(n,raw_data,idxs,batch_size,num_steps,n_use,next_n,FLAGS):
    VERBOSE=FLAGS.VERBOSE
    raw_x, raw_y = raw_data
    num_idxs=np.shape(idxs)[0]
    data = np.empty((num_idxs,num_steps+next_n,n_use))

    if VERBOSE:
        print('raw_data shape:',np.shape(raw_x))
        print('indexes_shape',np.shape(idxs))
        print('partioning raw_data...')
    for i,idx in enumerate(idxs):
        # Make a data matrix thats [num_idxs, num_steps+next_n, num_neurons]
        data[i,:,:] = np.reshape(raw_x[:, idx*num_steps:((idx+1)*num_steps)+next_n],[1,num_steps+next_n,n_use])
    if VERBOSE:
        print('data matrix shape',np.shape(data))

    # Pass the whole data matrix to gen batch for every epoch
    for i in range(n):
        yield gen_batch(data, idxs, batch_size, num_steps,n_use,next_n,FLAGS.VERBOSE)


class KohnUtils:

    def make_timeseries(kohn_data,kohn_stim_seq, latency=None):
        response = []
        # Filter out poor quality neurons
        mask = np.squeeze(kohn_data['INDCENT']).astype(bool)

        # Get recordings during stimulus
        resp_train = kohn_data['resp_train'][mask]
        # Get post-stim recordings
        resp_train_blk = mat_file['resp_train_blk'][mask]

        # Concatenate them making a complete trial timeseries
        resp = np.concatenate((resp_train,resp_train_blk), axis=3)

        if not latency==None:
            resp = np.roll(resp,-latency,3)[:,:,:,:-latency]

        # [neurons,image_id,trial_id]
        trials = np.size(resp,2)
        num_neurons = np.size(resp,0)
        num_images = np.size(resp, 1)
        i = 0
        for r in tqdm(np.arange(trials)):
            for image_id in stim_sequence[:,r]:
                index = {'i': i,
                         'trial': r,
                         'image': image_id-1
                         }
                #x_on = np.zeros(105, dtype=int) + 1
                #x_off= np.zeros(211, dtype=int) + 0
                #x = np.concatenate((x_on, x_off))

                y = resp[:,image_id-1, r,:]
                #sequences.extend([x])

                response.extend([y])
                i = i+1
                #print(index)
                #print(time_pts)
                #print(time_pts+316*i)
                #print(ms)
            #print(index)
            #print(x.shape)
            #print(x)
            #print(y.shape)
            #print(y)
        return np.concatenate(np.array(response)) #,np.concatenate(np.array(labels), axis=1))

    def get_timeseries(kohn_data,kohn_stim_seq):
        return make_timeseries(kohn_data, kohn_stim_seq)

    def save_timeseries(kohn_data,kohn_stim_seq,SAVE_PATH):
        ts = make_timeseries(kohn_data,kohn_stim_seq)
        sio.savemat(SAVE_PATH,{
            'timeseries':ts
                    })

