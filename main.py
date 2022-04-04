"""
This is the main module for running TextRecognizer
"""
import PIL.Image

from libraries import TextRecognizer

import os
import cv2
from PIL import Image

from configparser import ConfigParser


def run_sample_image():
    """
    Test class for running sample images
    """
    img = Image.open("sample-images/handwriting.png")
    TextRecognizer.run_image(img)


def run_sample_stream():
    """
    Test class for running a live stream of image parsing
    """
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    streamRTSP = os.environ.get("STREAM_RTSP")
    capture = cv2.VideoCapture(str(streamRTSP))
    # Purely for efficiency and not needing to read the config file each frame
    config = ConfigParser()
    config.read('libraries/config.ini')

    while True:
        _, frame = capture.read()
        frame = TextRecognizer.run_video(frame, config)
        cv2.imshow('livestream', frame)
        if cv2.waitKey(1) == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_sample_stream()
