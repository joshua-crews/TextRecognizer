"""Takes an ingest image and converts it to a binarized format"""
from PIL import Image


def binarize(image):
    """
    Function used to binarize an image
    :param image: PIL image that will be processed
    """
    image.show()
