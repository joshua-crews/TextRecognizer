"""Network analyzer that calculates the image in question, the meat of the code"""
import numpy
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageDraw, ImageFont

from configparser import ConfigParser

config = ConfigParser()
config.read('libraries/config.ini')

accuracy_threshold = float(config.get('Mapping_Options', 'accuracy_threshold'))

arr_out = []
arr_result = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g',
              'h', 'n', 'q', 'r', 't']

outputImage = None
isFirstOutput = True

model = load_model('libraries/network/model/hqm.h5')


def sortcnts(cnts):  # to sort the contours left to right

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][0], reverse=False))

    return (cnts)


def test(a, b, c, d, imd, createOutputImage):
    """
    Predicts the character present in the specified region of the image
    :param imd: Numpy array of image
    """
    test = imd[b:b + d, a:a + c]
    _, test_image = cv2.threshold(test, 100, 255, cv2.THRESH_BINARY)
    test_image = cv2.copyMakeBorder(test_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    test_image = cv2.medianBlur(test_image.copy(), 3)
    test_image = cv2.resize(test_image.copy(), (28, 28), interpolation=cv2.INTER_AREA)
    cv2.resize(test_image, (28, 28))
    test_image = np.invert(test_image)
    test_image = (image.img_to_array(test_image)) / 255
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    np.reshape(result, 47)
    high = np.amax(test_image)
    low = np.amin(test_image)
    if high != low:
        maxval = np.amax(result)
        if maxval > accuracy_threshold:
            index = np.where(result == maxval)
            arr_out.append(arr_result[index[1][0]])
            if createOutputImage is not False:
                global outputImage
                outputImage = 'output.png'
            return arr_result[index[1][0]]


def predict(img, boxColor, boxWidth, createOutput=False):
    img = np.array(img)
    blur = cv2.bilateralFilter(img.copy(), 9, 75, 75)
    _, thresh = cv2.threshold(blur.copy(), 200, 255, cv2.THRESH_BINARY)

    contours, h = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sum = 0
    maxar = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        sum += (w * h)

    avg = sum / len(contours)
    maxar = 10000
    minar = 1000
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < maxar and w * h > minar:
            char = test(x, y, w, h, img, createOutput)
            global outputImage
            if outputImage is not None:
                try:
                    global isFirstOutput
                    if isFirstOutput:
                        sampleOutput = np.array(createOutput)
                        isFirstOutput = False
                    else:
                        sampleOutput = np.array(Image.open(outputImage))
                except FileNotFoundError:
                    sampleOutput = np.array(createOutput)
                if char is not None:
                    cv2.rectangle(sampleOutput, (x, y), (x + w, y + h), boxColor, boxWidth)
                    sampleOutput = Image.fromarray(np.uint8(sampleOutput))
                    draw = ImageDraw.Draw(sampleOutput)
                    font = ImageFont.truetype("./arial.ttf", 30)
                    draw.text((x, y - (boxWidth * 4) - 20), str(char), fill=(255, 0, 0), font=font)
                    sampleOutput.save(outputImage, 'PNG')

    final = ''
    i = 0
    for ch in reversed(arr_out):
        i += 1
        final = final + ch

    if outputImage is not None:
        sampleOutput.show()

    return final


def predict_still(img, original_img, boxColor, boxWidth):
    """
    Used to predict characters in a live video frame at an accelerated but also less accurate rate
    :param img:
    :param original_img:
    :param boxColor:
    :param boxWidth:
    :return:
    """
    blur = cv2.bilateralFilter(img.copy(), 9, 75, 75)
    _, thresh = cv2.threshold(blur.copy(), 200, 255, cv2.THRESH_BINARY)

    contours, h = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sum = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        sum += (w * h)

    maxar = 10000
    minar = 1000
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if maxar > w * h > minar:
            char = test(x, y, w, h, img, False)
            if char is not None:
                original_img = numpy.array(original_img)
                cv2.rectangle(original_img, (x, y), (x + w, y + h), boxColor, boxWidth)
                original_img = Image.fromarray(np.uint8(original_img))
                draw = ImageDraw.Draw(original_img)
                font = ImageFont.truetype("./arial.ttf", 30)
                draw.text((x, y - (boxWidth * 4) - 20), str(char), fill=(255, 0, 0), font=font)

    try:
        return np.array(original_img)
    except UnboundLocalError:
        return img


cv2.waitKey()
cv2.destroyAllWindows()
