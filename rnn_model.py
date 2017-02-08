import tensorflow as tf

class RecurrentActivityModel:

    def __init__(self, x, y, x_mean, num_steps=50, state_size=25, next_n=1,learning_rate=1e-3, num_layers=3):

        # Config Variables
        self.x = x # [1,num_steps,n_use=state_size]
        self.y = y
        self.num_steps = num_steps  # number of truncated backprop steps ('n')
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        global_step = tf.Variable(0, name='global_step', trainable=False)
        batch_size = self.x.get_shape()[0]
        self.n_use = int(self.x.get_shape()[2])

        rnn_inputs = self.x
        weight = tf.get_variable('weight',[self.state_size, self.n_use], initializer=tf.constant_initializer(1.0))
        bias = tf.get_variable('bias', [self.n_use], initializer=tf.constant_initializer(0.1))

        cell = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)
        #cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers,state_is_tuple=True)
        self.init_state = cell.zero_state(batch_size, tf.float32)

        # rnn_outputs shape = (batch_size, num_steps, state_size)
        # final_state = rnn_outputs[:,-1,:] = (30, 25)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(
            cell,
            rnn_inputs,
            initial_state=self.init_state,
            dtype=tf.float32
        )
        # Flatten rnn_outputs down to shape = (batch_size*num_steps, state_size)
        last_out = rnn_outputs[:,-next_n:,:]
        out_mod = tf.reshape(rnn_outputs, [-1, state_size])
        # Flatten y; (num_steps,n_use) -> (num_steps*n_use)

        # (1500,25) x (25,2) = (1500,2)
        logits = tf.matmul(out_mod, weight) + bias
        logits_reshaped = tf.reshape(logits,[-1,num_steps,state_size])
        self._weight_matrix = weight
        #seqw =  tf.ones((batch_size, num_steps))
        self._prediction = logits_reshaped[:,-next_n:,:]
        #self._prediction = tf.reshape(self._flat_prediction, [-1, self.num_steps])
        #for i in range(self.n_use-1):
            #tf.summary.histogram('prediction_n%d'%(i),self._prediction[i,:])

        with tf.name_scope('error'):
            self._error = self._prediction-self.y
            with tf.name_scope('total_loss'):
                self._total_loss = tf.reduce_sum(tf.pow(self._error,2))

        #logits_1 = tf.reshape(logits, [-1, num_steps, num_classes])
        #seq_loss = tf.nn.seq2seq.sequence_loss_by_example(
        #    tf.unpack(logits_1, axis=1),
        #    tf.unpack(self.y, axis=1),
        #    tf.unpack(seqw, axis=1),
        #    average_across_timesteps=True
        #    #softmax_loss_function=
        #)
        #perplexity = tf.exp(seq_loss)
        #self._avg_perplexity = tf.reduce_mean(perplexity)
        tf.summary.scalar('total_loss', self._total_loss)
        #tf.summary.scalar('avg_perplexity', self._avg_perplexity)
        self._optimize = tf.train.AdamOptimizer(learning_rate).minimize(self._total_loss, global_step=global_step)
        self._y_mean = tf.reduce_mean(self.y,axis=1,keep_dims=True)

        with tf.name_scope('accuracy'):
             with tf.name_scope('correct_prediction'):
                #correct_prediction = self.prediction-self.y
                null_error = tf.zeros_like(self._prediction)-self.y
                mean_error = self._y_mean-self.y
             #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
             null_loss = tf.reduce_sum(tf.pow(null_error, 2))
             var = tf.reduce_sum(tf.pow(mean_error,2))
             self.FEV = 1-(self._total_loss/var)
        tf.summary.scalar('null_loss', null_loss)
        tf.summary.scalar('var', var)
        tf.summary.scalar('FEV', self.FEV)

        self._merge_summaries = tf.summary.merge_all()
        self._global_step = global_step

    def do(self, session, fetches, feed_dict):
        vals = session.run(fetches, feed_dict)

        return vals

    def step(self,session):
        return tf.train.global_step(session,self._global_step)

    @property
    def prediction(self):
        return self._prediction

    @property
    def optimize(self):
        return self._optimize

    @property
    def error(self):
        return self._error

    @property
    def avg_perplexity(self):
        return self._avg_perplexity

    @property
    def total_loss(self):
        return self._total_loss
    @property
    def merge_summaries(self):
        return self._merge_summaries

