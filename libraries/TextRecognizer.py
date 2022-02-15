"""Pass images to return a text result"""
from libraries.preprocessing import Binarizer
from libraries.network import Network

import os
from configparser import ConfigParser


def run(image):
    """
    Main function called to process an image
    :param image: PIL image that needs to be processed
    """
    # Setup config file
    config = ConfigParser()
    config.read('libraries/config.ini')

    # Process image
    binImage = Binarizer.binarize(image)
    boxColor = config.get('Mapping_Options', 'color')
    boxColor = eval(boxColor)
    text = Network.predict(binImage, boxColor, int(config.get('Mapping_Options', 'width')), image)
    print(f'The final value of text is: {text}')
