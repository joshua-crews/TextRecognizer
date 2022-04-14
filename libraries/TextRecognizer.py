"""Pass images to return a text result"""
import cv2

from libraries.preprocessing import Binarizer
from libraries.network import Network

import os
from PIL import Image
from configparser import ConfigParser


def run_image(image):
    """
    Main function called to process an image
    :param image: PIL image that needs to be processed
    """
    # Setup config file
    config = ConfigParser()
    config.read('libraries/config.ini')

    # Process image
    binarization_thesh = int(config.get('Mapping_Options', 'binarization_threshold'))
    binImage = Binarizer.binarize(image, binarization_thesh)
    boxColor = config.get('Mapping_Options', 'color')
    boxColor = eval(boxColor)
    text = Network.predict(img=binImage, boxColor=boxColor,
                           boxWidth=int(config.get('Mapping_Options', 'width')), createOutput=image)
    print(f'The final value of text is: {text}')


def run_video(frame, config=None):
    """
    Main function called to process a video or stream
    :param frame: A numpy array of the current frame in video
    :param config: A loaded configparser, not needed however increases efficiency if provided
    """

    # Create a config parser
    if config is None:
        config = ConfigParser()
        config.read('libraries/config.ini')

    # Binarize the frame
    im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(im_gray, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
    im_thresh_gray = cv2.bitwise_and(im_gray, mask)
    processedFrame = im_thresh_gray.copy()

    boxColor = config.get('Mapping_Options', 'color')
    boxColor = eval(boxColor)

    completeFrame = processedFrame
    completeFrame = Network.predict_still(img=processedFrame, original_img=frame, boxColor=boxColor, boxWidth=int(config.get('Mapping_Options', 'width')))
    return completeFrame
