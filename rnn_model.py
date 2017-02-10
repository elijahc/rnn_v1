import tensorflow as tf

class RecurrentActivityModel:

    def __init__(self, x, y, x_mean, num_steps=50, state_size=25, next_n=1,learning_rate=1e-4, num_layers=3):
        def squared(tensor):
            return tf.pow(tensor,2)

        def sse(error):
            return tf.reduce_sum(squared(error))


        # Config Variables
        self.x = x # [1,num_steps,n_use=state_size]
        self.y = y
        self.x_mean = x_mean
        self.num_steps = num_steps  # number of truncated backprop steps ('n')
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.y_mean = tf.reduce_mean(self.y,axis=1,keep_dims=True)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        batch_size = int(self.x.get_shape()[0])
        self.n_use = int(self.x.get_shape()[2])

        learning_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step,
                10000,
                0.80,
                staircase=False
                )

        with tf.variable_scope('out_weights'):
            W = tf.get_variable(
                    'W',
                    [self.state_size, self.n_use],
                    #initializer=tf.constant_initializer(1.0)
                    initializer=tf.contrib.layers.xavier_initializer()
                    )
            bias = tf.get_variable(
                    'bias',
                    [self.n_use],
                    initializer=tf.constant_initializer(0.1)
                    )

        self._weight_matrix = W
        self._learning_rate = learning_rate

        # Define RNN architecture
        cell = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers,state_is_tuple=True)
        self.init_state = cell.zero_state(batch_size, tf.float32)


        # Connect rnn_inputs to architecture defined above
        # rnn_outputs shape = (batch_size, num_steps, state_size)
        # final_state = rnn_outputs[:,-1,:] = (batch_size, state_size)
        rnn_inputs = self.x
        rnn_outputs, final_state = tf.nn.dynamic_rnn(
            cell,
            rnn_inputs,
            initial_state=self.init_state,
            dtype=tf.float32
        )

        # Grab last n values
        last_n_out = rnn_outputs[:,-next_n:,:]
        # Flatten rnn_outputs down to shape = (batch_size*num_steps, state_size)
        out_mod = tf.reshape(last_n_out, [-1, state_size])
        # Flatten y; (num_steps,n_use) -> (num_steps*n_use)
        _y      = tf.reshape(self.y, [-1,state_size])
        _y_mean = tf.reshape(self.y_mean, [-1,state_size])
        _x_mean = tf.reshape(self.x_mean, [-1,state_size])

        # (1500,25) x (25,2) = (1500,2)
        logits = tf.matmul(out_mod, W) + bias
        logits_reshaped = tf.reshape(logits,[batch_size,next_n,self.n_use])
        #seqw =  tf.ones((batch_size, num_steps))
        self._prediction = logits_reshaped
        #self._prediction = tf.reshape(self._flat_prediction, [-1, self.num_steps])
        #for i in range(self.n_use-1):
            #tf.summary.histogram('prediction_n%d'%(i),self._prediction[i,:])

        with tf.name_scope('metrics'):
            # Error Metrics
            with tf.name_scope('error'):
                self._error     = logits-_y

                with tf.name_scope('2d'):
                    self._error_2d   = self._prediction-self.y

            with tf.name_scope('null_error'):
                null_error = tf.zeros_like(logits)-_y
            with tf.name_scope('mean_error'):
                mean_error = self.y - self.x_mean
            with tf.name_scope('squared_error'):
                self.se         = squared(self._error)
                with tf.name_scope('2d'):
                    self.se_2d     = squared(self._error)

            # 1D Metrics
            with tf.name_scope('total_loss'):
                self._total_loss = sse(self._error)
                self.FEV_2d    = tf.reduce_sum(squared(self._error_2d),[0])

            with tf.name_scope('null_loss'):
                null_loss = sse(null_error)

            with tf.name_scope('variance'):
                var = sse(mean_error)
                self._var_2d = tf.reduce_sum(squared(mean_error),[0])

            with tf.name_scope('FEV'):
                self.FEV = 1-(self._total_loss/var)
                #self.FEV_2d = 1-(self._sse_2d/self._var_2d)

        self._optimize = tf.train.AdamOptimizer(learning_rate).minimize(self._total_loss, global_step=global_step)

        # Log outputs to summary writer
        tf.summary.scalar('total_loss', self._total_loss)
        tf.summary.scalar('null_loss', null_loss)
        tf.summary.scalar('var', var)
        tf.summary.scalar('FEV', self.FEV)
        #tf.summary.scalar('avg_perplexity', self._avg_perplexity)
        self._merge_summaries = tf.summary.merge_all()
        self._global_step = global_step

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
    def status(self):
        status = dict(
                prediction=self.prediction,
                lr=self._learning_rate
                )
        return status

    @property
    def total_loss(self):
        return self._total_loss
    @property
    def merge_summaries(self):
        return self._merge_summaries

