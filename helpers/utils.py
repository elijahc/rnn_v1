import numpy as np

def gen_batch(raw_data, idxs, lookups, FLAGS):
    batch_size = FLAGS.batch
    num_steps = FLAGS.seq_len
    n_use = FLAGS.n_use
    next_n = FLAGS.guess
    VERBOSE = FLAGS.VERBOSE
    #raw_x = np.squeeze(raw_x)
    #raw_y = np.squeeze(raw_y)

    # partition raw data into batches and stack them vertically in a data matrix
    num_idxs = np.shape(idxs)[0]
    batch_partition_length = num_idxs // batch_size
    batch_idxs = np.random.permutation(num_idxs)

    for i in np.arange(batch_partition_length):
        xy = raw_data[i*batch_size:(i+1)*batch_size]
        x = xy[:,:num_steps]
        y = xy[:,-next_n:]
        yield (x,y,xy,lookups)


def gen_epochs(n,raw_data,idxs,FLAGS):
    num_steps = FLAGS.seq_len
    n_use = FLAGS.n_use
    next_n= FLAGS.guess
    batch_size= FLAGS.batch

    VERBOSE=FLAGS.VERBOSE
    raw_x = raw_data
    num_idxs=np.shape(idxs)[0]
    data = np.empty((num_idxs,num_steps+next_n))
    # Map ints to tuples
    n_vec_set = set()
    vec_to_id={}
    id_to_vec=[]
    next_key=0

    if VERBOSE:
        print('raw_data shape:',np.shape(raw_x))
        print('indexes_shape',np.shape(idxs))
        print('partioning raw_data...')
    for i,idx in enumerate(idxs):
        # Make a data matrix thats [num_idxs, num_steps+next_n, num_neurons]
        ex = np.reshape(raw_x[:, idx*num_steps:((idx+1)*num_steps)+next_n],[1,num_steps+next_n,n_use])
        for step in np.arange(num_steps+next_n):
            ex_tup = tuple(ex[0,step].tolist())
            if not ex_tup in n_vec_set:
                vec_to_id[ex_tup] = next_key
                id_to_vec.extend([ex_tup])
                next_key = next_key+1
                n_vec_set.add(ex_tup)

            # Lookup ID
            data[i,step] = vec_to_id[ex_tup]

    lookups = dict(
            id_to_vec=id_to_vec,
            vec_to_id=vec_to_id,
            n_vec_set=n_vec_set)

    # Pass the whole data matrix to gen batch for every epoch
    for i in range(n):
        yield gen_batch(data, idxs, lookups,FLAGS)

def vec_set(raw_data,FLAGS,idxs):
    num_idxs=np.ma.size(idxs,axis=-1)

    uniq_input = set()
    for epoch in gen_epochs(1,raw_data,idxs,FLAGS):
        for i,(x,y,xy,lookups) in enumerate(epoch):
            return lookups['n_vec_set']



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

