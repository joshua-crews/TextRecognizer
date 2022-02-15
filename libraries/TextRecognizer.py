"""Pass images to return a text result"""
from libraries.preprocessing import Binarizer
from libraries.network import Network


def run(image):
    """
    Main function called to process an image
    :param image: PIL image that needs to be processed
    """
    binImage = Binarizer.binarize(image)
    # binImage.show()
    text = Network.predict(binImage, True)
    print(f'The final value of text is: {text}')
