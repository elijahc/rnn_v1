import tensorflow as tf

class RecurrentActivityModel:

    def __init__(self, x, y, x_mean, x_set_dim, y_set_dim, FLAGS):
        def squared(tensor):
            return tf.pow(tensor,2)

        def sse(error):
            return tf.reduce_sum(squared(error))


        # Config Variables
        self.x = x # [1,num_steps,n_use=state_size]
        self.y = y
        self.x_mean = x_mean
        self.num_steps = FLAGS.seq_len  # number of truncated backprop steps ('n')
        self.learning_rate = FLAGS.lr
        self.y_mean = tf.reduce_mean(self.y,axis=1,keep_dims=True)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        batch_size = FLAGS.batch
        self.n_use = FLAGS.n_use
        self.vocab_size = x_set_dim
        self.resp_size  = y_set_dim
        next_n = FLAGS.guess

        learning_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step,
                10000,
                0.80,
                staircase=False
                )

        embedding = tf.get_variable(
                    "embedding", [self.vocab_size,FLAGS.rnn_size],dtype=tf.float32)

        y_OH = tf.one_hot(self.y,self.resp_size)

        rnn_inputs = tf.nn.embedding_lookup(embedding,self.x)

        with tf.variable_scope('out_weights'):
            W = tf.get_variable(
                    'W',
                    [FLAGS.rnn_size, self.n_use],
                    #initializer=tf.constant_initializer(1.0)
                    initializer=tf.contrib.layers.xavier_initializer()
                    )
            bias = tf.get_variable(
                    'bias',
                    [self.n_use],
                    initializer=tf.constant_initializer(0.1)
                    )

        with tf.variable_scope('weight2'):
            W2 = tf.get_variable(
                    'W2',
                    [FLAGS.batch*self.resp_size,FLAGS.batch*FLAGS.seq_len]
                    )
            b2 = tf.get_variable(
                    'b2',
                    [self.n_use],
                    initializer=tf.constant_initializer(0.1)
                    )
        self._weight_matrix = W

        # Define RNN architecture
        cell = tf.nn.rnn_cell.LSTMCell(FLAGS.rnn_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * FLAGS.layers,state_is_tuple=True)
        self.init_state = cell.zero_state(FLAGS.batch, tf.float32)


        # Connect rnn_inputs to architecture defined above
        # rnn_outputs shape = (batch_size, num_steps, state_size)
        # final_state = rnn_outputs[:,-1,:] = (batch_size, state_size)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(
            cell,
            rnn_inputs,
            initial_state=self.init_state,
            dtype=tf.float32
        )

        # Grab last n values
        #last_n_out = rnn_outputs[:,-next_n:,:]
        # Flatten rnn_outputs down to shape = (batch_size*num_steps, state_size)
        out_mod = tf.reshape(rnn_outputs, [-1, FLAGS.rnn_size])

        logits = tf.matmul(out_mod, W) + bias

        _response_vec = tf.matmul(W2,logits) + b2
        response_vec = tf.reshape(_response_vec, [FLAGS.batch,FLAGS.n_use,self.resp_size])

        # Flatten y; (num_steps,n_use) -> (num_steps*n_use)
        #_y_mean = tf.reshape(self.y_mean, [-1,FLAGS.rnn_size])
        #_x_mean = tf.reshape(self.x_mean, [-1,FLAGS.n_use])

        # (1500,25) x (25,2) = (1500,2)
        #seqw =  tf.ones((batch_size, num_steps))

        ###### Stopping point
        self._prediction = tf.argmax(response_vec,axis=2)
        #self._prediction = tf.reshape(self._flat_prediction, [-1, self.num_steps])
        #for i in range(self.n_use-1):
            #tf.summary.histogram('prediction_n%d'%(i),self._prediction[i,:])

        import pdb; pdb.set_trace()
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

