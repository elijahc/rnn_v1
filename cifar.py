from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

batch_size  = 32
nb_classes  = 10
nb_epoch    = 200
data_augmentation = False

# Data shuffled and split between train and test sets:
(X_train,y_train),(X_test,y_test) = cifar10.load_data()

Y_train = np_utils.to_categorical(y_train,nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Define model
model = Sequential()


model.add(Convolution2D(32,3,3, border_mode='same',
                        input_shape=X_train.shape[1:]))

model.add(Activation('relu'))
model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64,3,3,border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Train using RMSprop
model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train,Y_train,
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            validation_data=(X_test,Y_test),
            shuffle=True)
