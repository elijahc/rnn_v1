# TODO:
1. Rework model to operate off of continuous data
    1. Change data sampling
        - Re-sample/bin data from 1ms increments to 10ms increments
        - Sum spikes for the 10ms interval
        - Use as input to prediction algorithm
    2. Change prediction algorithm
        - Figure out how to mutate input data into form tf.nn.dyanmic_rnn will accept
            - [batch_size x num_timesteps x state_size]
