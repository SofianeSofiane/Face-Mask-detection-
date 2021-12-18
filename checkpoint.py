from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Flatten,Dropout
from tensorflow.keras.layers import Conv2D,MaxPooling2D


my_model=Sequential()

my_model.add(Conv2D(64,(3,3),input_shape=(200, 200, 1)))
my_model.add(Activation('relu'))
my_model.add(MaxPooling2D(pool_size=(2,2)))

my_model.add(Conv2D(128,(3,3)))
my_model.add(Activation('relu'))
my_model.add(MaxPooling2D(pool_size=(2,2)))

my_model.add(Flatten())
my_model.add(Dropout(0.5))

my_model.add(Dense(64,activation='relu'))

my_model.add(Dense(2,activation='softmax'))

import os

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

import tensorflow as tf
latest = tf.train.latest_checkpoint(checkpoint_dir)

my_model.load_weights(checkpoint_path)

print(my_model.summary())

my_model.save('mask_detection.model')