"""Pass images to return a text result"""
from libraries.preprocessing import Binarizer


def run(image):
    """
    Main function called to process an image
    :param image: PIL image that needs to be processed
    """
    binImage = Binarizer.binarize(image)
