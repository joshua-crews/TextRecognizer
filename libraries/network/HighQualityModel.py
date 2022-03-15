import logging

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from extra_keras_datasets import emnist

import matplotlib.pyplot as plt
from configparser import ConfigParser

config = ConfigParser()
config.read('../config.ini')

network_visualizer = bool(config.get('Modelling_Options', 'network_visualizer'))
if network_visualizer:
    from ann_visualizer.visualize import ann_viz
    # This is required to use graphviz in a local environment
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

epochs = int(config.get('Modelling_Options', 'training_epochs'))
batch_size = int(config.get('Modelling_Options', 'batch_size'))

num_classes = 47
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = emnist.load_data(type='balanced')

# Visualize data
f, ax = plt.subplots(1, num_classes, figsize=(20, 20))
arr_result = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g',
              'h', 'n', 'q', 'r', 't']
for i in range(0, num_classes):
  sample = x_train[y_train == i][0]
  ax[i].imshow(sample, cmap='gray')
  ax[i].set_title("{}".format(arr_result[i]), fontsize=16)
if network_visualizer:
    plt.show()

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size=4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

if network_visualizer:
    try:
        ann_viz(model, title="High Quality Model View")
    except IndexError:
        logging.warning('Could not complete network model visualizer, unknown layer type being used!')

model.save('model/hqm.h5')
