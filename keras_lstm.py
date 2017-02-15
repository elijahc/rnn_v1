import numpy
import matplotlib.pyplot as plt
import math
import scipy.io as sio
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

numpy.random.seed(7)

mat_file = sio.loadmat('data/timeseries/10_timeseries.mat')


model = Sequential()
model.add(LSTM(4, batch_input_shape=(batch_size, time_steps, features ), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX,trainY,nb_epoch=100,batch_size=1,verbose=2)


