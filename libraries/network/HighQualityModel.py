"""
import numpy as np
import cv2
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing import image
import scipy.fftpack

import HighQualityModelUtils as HQMUtils
from configparser import ConfigParser

config = ConfigParser()
config.read('../config.ini')

batch = int(config.get('Modelling_Options', 'batch_ratio'))
epochs = int(config.get('Modelling_Options', 'training_epochs'))
steps = int(config.get('Modelling_Options', 'steps_per_epoch'))
valid_steps = int(config.get('Modelling_Options', 'valid_epochs'))

network_visualizer = bool(config.get('Modelling_Options', 'network_visualizer'))
if network_visualizer:
    from ann_visualizer.visualize import ann_viz
    # This is required to use graphviz in a local environment
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

arr_result = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z']

input_shape = (20, 20, 1)

model = Sequential()
model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Dropout(0.5))
model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(40, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(40, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=47, activation='sigmoid'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

training_set, test_set = HQMUtils.get_datasets(batch)

model.fit(training_set, steps_per_epoch=steps, epochs=epochs, validation_data=test_set, validation_steps=valid_steps)

if network_visualizer:
    ann_viz(model, title="High Quality Model View")

model.save('model/hqm.h5')
"""

"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import cv2

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import sklearn.metrics as metrics

from configparser import ConfigParser

config = ConfigParser()
config.read('../config.ini')

epochs = int(config.get('Modelling_Options', 'training_epochs'))
steps = int(config.get('Modelling_Options', 'steps_per_epoch'))

network_visualizer = bool(config.get('Modelling_Options', 'network_visualizer'))
if network_visualizer:
    from ann_visualizer.visualize import ann_viz
    # This is required to use graphviz in a local environment
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

train = pd.read_csv(config.get('Modelling_Options', 'dataset_path_training'), delimiter=',')
test = pd.read_csv(config.get('Modelling_Options', 'dataset_path_validation'), delimiter=',')
mapp = pd.read_csv(config.get('Modelling_Options', 'mapping_path'), delimiter=' ', index_col=0, header=None).squeeze(
    "columns")

HEIGHT = 28
WIDTH = 28
train_x = train.iloc[:, 1:]
train_y = train.iloc[:, 0]
del train

test_x = test.iloc[:, 1:]
test_y = test.iloc[:, 0]
del test


def rotate(image):
    image = image.reshape([HEIGHT, WIDTH])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image


train_x = np.asarray(train_x)
train_x = np.apply_along_axis(rotate, 1, train_x)

test_x = np.asarray(test_x)
test_x = np.apply_along_axis(rotate, 1, test_x)

train_x = train_x.astype('float32')
train_x /= 255
test_x = test_x.astype('float32')
test_x /= 255

for i in range(100, 109):
    plt.subplot(330 + (i + 1))
    plt.imshow(train_x[i], cmap=plt.get_cmap('gray'))
    plt.title(chr(mapp[train_y[i]]))

num_classes = train_y.nunique()
train_y = np_utils.to_categorical(train_y, num_classes)
test_y = np_utils.to_categorical(test_y, num_classes)

train_x = train_x.reshape(-1, HEIGHT, WIDTH, 1)
test_x = test_x.reshape(-1, HEIGHT, WIDTH, 1)

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.10, random_state=7)
model = Sequential()

model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(HEIGHT, WIDTH, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(units=num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_x, train_y, epochs=epochs, batch_size=steps, verbose=1, validation_data=(val_x, val_y))

if network_visualizer:
    ann_viz(model, title="High Quality Model View")

model.save('model/hqm.h5')"""

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
    ann_viz(model, title="High Quality Model View")

model.save('model/hqm.h5')
