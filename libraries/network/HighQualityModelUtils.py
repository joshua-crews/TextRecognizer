"""Builds the datasets for training and validation in model construction"""
from tensorflow.keras.datasets import mnist
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

from configparser import ConfigParser
import os


def get_datasets(batch):
    """
    Returns two (x,y) datasets of all 62 characters
    :param batch: How many batches to run the training on
    :return:
    """
    config = ConfigParser()
    config.read('../config.ini')

    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_data = config.get('Modelling_Options', 'dataset_path_training')
    validation_data = config.get('Modelling_Options', 'dataset_path_validation')

    training_set = train_datagen.flow_from_directory(directory=training_data,
                                                     target_size=(20, 20),
                                                     color_mode='grayscale',
                                                     batch_size=batch,
                                                     class_mode='sparse')

    test_set = test_datagen.flow_from_directory(directory=validation_data,
                                                target_size=(20, 20),
                                                color_mode='grayscale',
                                                batch_size=batch,
                                                class_mode='sparse')
    return training_set, test_set
