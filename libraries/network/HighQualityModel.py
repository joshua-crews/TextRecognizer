"""import cv2
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import HighQualityModelUtils as ResNet

EPOCHS = 2
INIT_LR = 1e-1
BS = 128

(azData, azLabels) = ResNet.load_az_dataset('../../training-dataset2/Dataset.csv')
(digitsData, digitsLabels) = ResNet.load_zero_nine_dataset()

azLabels += 10


data = np.vstack([azData, digitsData])
labels = np.hstack([azLabels, digitsLabels])

# each image in the A-Z and MNIST digts datasets are 28x28 pixels;
# however, the architecture we're using is designed for 32x32 images,
# so we need to resize them to 32x32
data = [cv2.resize(image, (32, 32)) for image in data]
data = np.array(data, dtype="float32")

data = np.expand_dims(data, axis=-1)
data /= 255.0

le = LabelBinarizer()

labels = le.fit_transform(labels)

classTotals = labels.sum(axis=0)
classWeight = {}

for i in range(0, len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=None, random_state=42)

aug = ImageDataGenerator(rotation_range=10,
                         zoom_range=0.05,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.15,
                         horizontal_flip=False,
                         fill_mode="nearest")

opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = ResNet.ResNet.build(32, 32, 1, len(le.classes_), (3, 3, 3),
                     (64, 64, 128, 256), reg=0.0005)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS,
    class_weight=classWeight,
    verbose=1)

predictions = model.predict(testX, batch_size=BS)
model.save('model/hqm.h5')"""

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

arr_result = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
              'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

"""model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(20, 20, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=52, activation='sigmoid'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])"""
input_shape = (20, 20, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(units=52, activation='sigmoid'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

training_set, test_set = HQMUtils.get_datasets(batch)

model.fit(training_set, steps_per_epoch=500, epochs=epochs, validation_data=test_set, validation_steps=21)

model.save('model/hqm.h5')

