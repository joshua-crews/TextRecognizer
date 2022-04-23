"""Takes an ingest image and converts it to a binarized format"""
from PIL import Image
import numpy


def binarize(image, threshold=120):
    """
    Function used to binarize an image
    :param image: PIL image that will be processed
    """
    img = image.convert('L')  # convert image to monochrome
    img = numpy.array(img)  # convert pixels to a 2x2 array
    img = __binarize_array(img, threshold)
    im = Image.fromarray(numpy.uint8(img))
    return im


def __binarize_array(numpy_array, threshold):
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                # Convert a pixel to white within the threshold
                numpy_array[i][j] = 255
            else:
                # Convert a pixel to black outside the threshold
                numpy_array[i][j] = 0
    return numpy_array
