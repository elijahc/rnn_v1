import numpy as np

def gen_batch(raw_data, t_vec,n_vec,idxs, FLAGS):
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
    t_maj = [FLAGS.batch,FLAGS.seq_len,FLAGS.n_use]
    t_map = {}
    n_map = {}

    for tid,vec in enumerate(list(t_vec)):
        t_map[vec] = tid

    for nid,vec in enumerate(list(n_vec)):
        n_map[vec] = nid

    lookups = dict(t_map=t_map,n_map=n_map)
    for i in np.arange(batch_partition_length):
        xy = raw_data[i*batch_size:(i+1)*batch_size].astype(int)
        x_out = np.empty((batch_size,num_steps))
        y_out = np.empty((batch_size,n_use))
        for b in np.arange(batch_size):
            for t in np.arange(num_steps):
                x_out[b,t] = t_map[tuple(xy[b,t,:].tolist())]
            for n in np.arange(n_use):
                y_out[b,n] = n_map[tuple(xy[b,-next_n:,n].tolist())]

        yield (x_out,y_out,xy,lookups)


def gen_epochs(n,raw_data,idxs,FLAGS):
    num_steps = FLAGS.seq_len
    n_use = FLAGS.n_use
    next_n= FLAGS.guess
    batch_size= FLAGS.batch
    VERBOSE=FLAGS.VERBOSE

    raw_x = raw_data
    num_idxs=np.shape(idxs)[0]
    data = np.empty((num_idxs,num_steps+next_n,n_use))
    t_vec = set()
    n_vec = set()

    if VERBOSE:
        print('raw_data shape:',np.shape(raw_x))
        print('indexes_shape',np.shape(idxs))
        print('partioning raw_data...')
    for i,idx in enumerate(idxs):
        # Make a data matrix thats [num_idxs, num_steps+next_n, num_neurons]
        ex = np.reshape(raw_x[:, idx*num_steps:((idx+1)*num_steps)+next_n],[1,num_steps+next_n,n_use])
        data[i] = ex
        x_ex = ex[:,:num_steps]
        y_ex = ex[:,-next_n:]
        for t in np.arange(num_steps):
            t_vec.add(tuple(x_ex[0,t,:].tolist()))
        for n in np.arange(n_use):
            n_vec.add(tuple(y_ex[0,:,n].tolist()))


        #x_sparse,x_maps = t_vec(x,FLAGS)
        #y_sparse,y_maps = t_vec(np.reshape(y,[FLAGS.batch,FLAGS.n_use,FLAGS.guess]),FLAGS)

    # Pass the whole data matrix to gen batch for every epoch
    for i in range(n):
        yield gen_batch(data,t_vec,n_vec,idxs,FLAGS)

def t_vec(data,FLAGS):
    # Input: [batch,num_steps,n_use]
    # Output: [batch,num_steps] int -> vec representation of each time step

    dim_len = np.size(data,axis=1)
    out = np.empty((FLAGS.batch,dim_len))
    # Map ints to tuples
    vec_set = set()
    vec_to_id={}
    id_to_vec=[]
    next_key=0

    for b in np.arange(FLAGS.batch):
        for step in np.arange(dim_len):
            ex_tup = tuple(data[b,step,:].tolist())
            if not ex_tup in vec_set:
                vec_to_id[ex_tup] = next_key
                id_to_vec.extend([ex_tup])
                next_key = next_key+1
                vec_set.add(ex_tup)

            # Lookup ID
            out[b,step] = vec_to_id[ex_tup]

    maps = dict(
            id_to_vec=id_to_vec,
            vec_to_id=vec_to_id,
            vec_set=vec_set)

    return (out,maps)



def vec_set(raw_data,FLAGS,idxs):
    num_idxs=np.ma.size(idxs,axis=-1)

    for epoch in gen_epochs(1,raw_data,idxs,FLAGS):
        for i,(x,y,xy,lookups) in enumerate(epoch):
            return lookups



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

