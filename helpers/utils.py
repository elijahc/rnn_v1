import numpy as np

def gen_batch(raw_data, idxs, batch_size, num_steps, n_use,next_n):
    raw_x, raw_y = raw_data
    data_length = np.size(raw_x,axis=-1)
    #raw_x = np.squeeze(raw_x)
    #raw_y = np.squeeze(raw_y)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = np.size(idxs) // batch_size
    batch_idxs = np.random.choice(idxs,(batch_size))

    shape = (batch_size, n_use,num_steps)
    ashape = [batch_size,num_steps,n_use]
    x = np.empty(shape)
    y = np.empty((batch_size,n_use,num_steps+next_n))
    xy= np.empty((batch_size,n_use,num_steps+next_n))

    for p in np.arange(batch_partition_length):
        for b in np.arange(batch_size):
            num_idxs = np.shape(idxs)[0]
            #rand_idxs = idxs[np.random.randint(batch_size)]
            for i in idxs:
                xy[b] = raw_x[:, i*num_steps:((i+1)*num_steps)+next_n]

            new_xy= np.reshape(xy,[batch_size,num_steps+next_n,n_use])
            new_x = new_xy[:,:-next_n,:]
            new_y = new_xy[:,-next_n:,:]
            yield (new_x,new_y,new_xy)
def gen_epochs(n,raw_data,idxs,batch_size,num_steps,n_use,next_n):
    for i in range(n):
        yield gen_batch(raw_data, idxs, batch_size, num_steps,n_use,next_n)


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

