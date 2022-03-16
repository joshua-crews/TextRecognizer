"""
This is the main module for running TextRecognizer
"""
from libraries import TextRecognizer

import os
import cv2
from PIL import Image
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

streamRTSP = os.environ.get("STREAM_RTSP")


def run_sample_image():
    """
    Test class for running sample images
    """
    img = Image.open("sample-images/alphabet2.png")
    TextRecognizer.run(img)


def run_sample_stream():
    """
    Test class for running a live stream of image parsing
    """
    capture = cv2.VideoCapture(str(streamRTSP))
    while True:
        _, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cv2.imshow('livestream', gray)
        if cv2.waitKey(1) == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_sample_stream()
