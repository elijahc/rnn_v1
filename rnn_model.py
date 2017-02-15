import tensorflow as tf

class RecurrentActivityModel:

    def __init__(self, x, y, x_mean, FLAGS):
        def squared(tensor):
            return tf.pow(tensor,2)

        def mse(error):
            return tf.reduce_mean(squared(error))

        def sse(error):
            return tf.reduce_sum(squared(error))


        # Config Variables
        STIM_CUE = FLAGS.STIM_CUE
        self.x = x # [1,num_steps,n_use=state_size]
        self.y = y
        self.x_mean = x_mean
        self.num_steps = FLAGS.num_steps  # number of truncated backprop steps ('n')
        self.state_size = FLAGS.state_size
        self.learning_rate = FLAGS.lr
        self.num_layers = int(FLAGS.num_layers)
        self.y_mean = tf.reduce_mean(self.y,axis=1,keep_dims=True)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        batch_size = FLAGS.batch_size
        self.n_use = FLAGS.n_use
        next_n = FLAGS.next_n

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
                    [self.n_use, self.n_use],
                    initializer=tf.constant_initializer(1.0)
                    #initializer=tf.contrib.layers.xavier_initializer()
                    )
            bias = tf.get_variable(
                    'bias',
                    [self.n_use],
                    initializer=tf.constant_initializer(0.1)
                    )

        self._weight_matrix = W
        self._learning_rate = learning_rate

        # Define RNN architecture
        with tf.variable_scope('back_rnn'):
            cell = tf.nn.rnn_cell.LSTMCell(
                    self.state_size,
                    forget_bias=10,
                    num_proj=self.n_use,
                    use_peepholes=True,
                    state_is_tuple=True)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers,state_is_tuple=True)
            self.init_state = cell.zero_state(batch_size, tf.float32)


            # Connect rnn_inputs to architecture defined above
            # rnn_outputs shape = (batch_size, num_steps, state_size)
            # final_state = rnn_outputs[:,-1,:] = (batch_size, state_size)
            rnn_inputs = self.x
            rnn_outputs, final_state = tf.nn.dynamic_rnn(
                cell,
                rnn_inputs,
                initial_state=self.init_state,
            )

        # Grab last n values
        #last_n_out = rnn_outputs[:,-next_n:,:]
        # Flatten rnn_outputs down to shape = (batch_size*num_steps, state_size)
        out_mod = tf.reshape(rnn_outputs, [-1, self.n_use])
        # Flatten y; (num_steps,n_use) -> (num_steps*n_use)
        #_y      = tf.reshape(self.y, [-1,self.n_use])
        #_y_mean = tf.reshape(self.y_mean, [-1,self.state_size])
        #_x_mean = tf.reshape(self.x_mean, [-1,self.state_size])

        # (1500,25) x (25,2) = (1500,2)
        logits = tf.matmul(out_mod, W) + bias


        logits_reshaped = tf.reshape(logits,[-1,self.num_steps,self.n_use])
        #seqw =  tf.ones((batch_size, num_steps))
        #self._prediction = tf.nn.softmax(logits_reshaped[:,:next_n])
        self._prediction = logits_reshaped[:,:next_n,:]

        #    self._prediction = self._prediction[:,:,:-1]
        #    self.y = self.y[:,:,:-1]
        #    self.y_mean = self.y_mean[:,:,:-1]
        #    self.x_mean = self.x_mean[:,:,1:]

        #self.y_OH = tf.one_hot(tf.cast(self.y,tf.int32),10)
        #self.x_mean_OH = tf.one_hot(tf.cast(self.x_mean,tf.int32),10)
        #self._prediction = tf.reshape(self._flat_prediction, [-1, self.num_steps])
        #for i in range(self.n_use-1):
            #tf.summary.histogram('prediction_n%d'%(i),self._prediction[i,:])

        with tf.name_scope('metrics'):
            # Error Metrics
            with tf.name_scope('error'):
                self._error     = self._prediction - self.y

            with tf.name_scope('null_error'):
                null_error = tf.zeros_like(self._prediction) - self.y
            with tf.name_scope('mean_error'):
                self.mean_error = self.y - self.x_mean
            with tf.name_scope('squared_error'):
                self.se         = squared(self._error)
                with tf.name_scope('2d'):
                    self.se_2d     = squared(self._error)

            if FLAGS.STIM_CUE:
                self._error = self._error[:,:,:-1]
                self.mean_error = self.mean_error[:,:,:-1]

            # 1D Metrics
            with tf.name_scope('total_loss'):
                self._total_loss = mse(self._error)
                #self._total_loss = tf.nn.softmax_cross_entropy_with_logits()
                #self.FEV_2d    = tf.reduce_sum(squared(self._error_2d),[0])

            with tf.name_scope('variance'):
                self.var = mse(self.mean_error)
                self._var_2d = tf.reduce_sum(squared(self.mean_error),[0])

            with tf.name_scope('FEV'):
                self.FEV = 1-(self._total_loss/self.var)
                #self.FEV_2d = 1-(self._sse_2d/self._var_2d)

        self._optimize = tf.train.AdamOptimizer(learning_rate).minimize(self._total_loss, global_step=global_step)

        # Log outputs to summary writer
        tf.summary.scalar('total_loss', self._total_loss)
        tf.summary.scalar('var', self.var)
        tf.summary.scalar('FEV', self.FEV)
        #tf.summary.scalar('avg_perplexity', self._avg_perplexity)
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
    def status(self):
        status = dict(
                prediction=self.prediction,
                lr=self._learning_rate,
                error=self._error,
                loss=self._total_loss,
                mean_error=self.mean_error,
                var=self.var,
                input_vals=self.x,
                y_mean=self.y_mean,
                true_vals = self.y)
        return status

    @property
    def total_loss(self):
        return self._total_loss
    @property
    def merge_summaries(self):
        return self._merge_summaries

