import numpy as np
import sys,ipdb,traceback

def info(type,value,tb):
    traceback.print_exception(type,value,tb)
    print
    ipdb.pm()

def gen_batch(raw_data, idxs,FLAGS):
    #raw_x = np.squeeze(raw_x)
    #raw_y = np.squeeze(raw_y)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_size = FLAGS.batch_size
    num_steps = FLAGS.num_steps
    next_n = FLAGS.next_n
    num_idxs = np.shape(idxs)[0]
    batch_partition_length = num_idxs // FLAGS.batch_size
    batch_idxs = np.random.permutation(num_idxs)

    for i in np.arange(batch_partition_length):
        xy = raw_data[i*batch_size:(i+1)*batch_size]
        x = xy[:,:num_steps,:]
        y = xy[:,-next_n:,:]
        yield (x,y,xy)


def gen_epochs(n,raw_data,idxs,FLAGS):
    VERBOSE=FLAGS.VERBOSE
    batch_size = FLAGS.batch_size
    num_steps = FLAGS.num_steps
    next_n = FLAGS.next_n
    n_use = FLAGS.n_use

    raw_x = raw_data
    num_idxs=np.shape(idxs)[0]
    data = np.empty((num_idxs,num_steps+next_n,n_use))

    if VERBOSE:
        print('raw_data shape:',np.shape(raw_x))
        print('indexes_shape',np.shape(idxs))
        print('partioning raw_data...')
    for i,idx in enumerate(idxs):
        # Make a data matrix thats [num_idxs, num_steps+next_n, num_neurons]
        if ((idx+1)*num_steps) + next_n < np.size(raw_x,axis=1):
            t_slice = np.swapaxes(raw_x[:, idx*num_steps:((idx+1)*num_steps)+next_n],0,1)
            data[i,:,:] = t_slice

    if VERBOSE:
        print('data matrix shape',np.shape(data))

    # Pass the whole data matrix to gen batch for every epoch
    for i in range(n):
        yield gen_batch(data, idxs,FLAGS)

def t_vec(data,FLAGS):
    # Input: [batch,num_steps,n_use]
    # Output: [batch,num_steps] int -> vec representation of each time step

    dim_len = np.size(data,axis=1)
    out = np.empty((FLAGS.batch,dim_len))
    # Map ints to tuples
    vmap = VecMap()

    for b in np.arange(FLAGS.batch):
        for step in np.arange(dim_len):
            ex_tup = tuple(data[b,step,:].tolist())
            vmap.add(ex_tup)

            # Lookup ID
            out[b,step] = vmap.id(ex_tup)

    return (out,maps)

def vec_set(raw_data,FLAGS,idxs):
    num_idxs=np.ma.size(idxs,axis=-1)

    for epoch in gen_epochs(1,raw_data,idxs,FLAGS):
        for i,(x,y,xy,lookups) in enumerate(epoch):
            return lookups

def set_map(s,FLAGS):

    out_map = {}
    for i,vec in enumerate(list(s)):
        out_map[vec] = i

    return out_map


class DataContainer():

    def __init__(self, raw_data, idxs, FLAGS):
        self.raw_data = raw_data
        self.FLAGS = FLAGS
        self.idxs = idxs


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

if __name__ == '__main__':
    import doctest
    doctest.testmod()
