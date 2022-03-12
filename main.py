"""
This is the main module for running TextRecognizer
"""
from libraries import TextRecognizer

from PIL import Image


def run_sample_image():
    """
    Test class for running sample images
    """
    img = Image.open("sample-images/test2.png")
    TextRecognizer.run(img)


if __name__ == '__main__':
    run_sample_image()
