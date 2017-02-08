import scipy.io as sio


class MatLoader():
    __stim_seq_path__ = 'data/stimulus_sequence.mat'

    def __init__(self):
        self._stim_seq_path = __stim_seq_path__

    def load_mat(MAT_PATH, STIM_SEQ_PATH=__stim_seq_path__):

        # Load files
        print('loading...stim_sequence')
        stim_sequence = sio.loadmat(STIM_SEQ_PATH)['stimulus_sequence']
        print('loading...matfile')
        mat_file = sio.loadmat(MAT_PATH)

        mat_file['stim_sequence'] = stim_sequence
        return mat_file

    # function for munging data
    def munge(self,MAT_PATH,STIM_SEQ_PATH=__stim_seq_path__):
        mat_file = self.load_mat(MAT_PATH, STIM_SEQ_PATH)
        yield (mat_file,stim_seq)

    # Function for munging and then saving data
    def munge_and_save(self,MAT_PATH, SAVE_PATH, STIM_SEQ_PATH=__stim_seq_path__):
        mat_file = self.load_mat(MAT_PATH,STIM_SEQ_PATH)
        yield mat_file
        save_mat(SAVE_PATH, mat_file)

    def save_mat(SAVE_PATH, DATA):
        print('writing ', SAVE_PATH, '...')
        sio.savemat(SAVE_PATH, DATA)

class Munge():
    def __init__(self,filename):
        self.load_mat = MatLoader.load_mat
        self.filename=filename

    def __enter__(self):
        self.matfile = self.load_mat(self.filename)
        return self.matfile

    def __exit__(self):
        #do nothing
        print('exiting')
