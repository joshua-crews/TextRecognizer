"""Network analyzer that calculates the image in question, the meat of the code"""
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageDraw, ImageFont

arr_out = []
arr_result = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
"""arr_result = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
              'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']"""

outputImage = None
isFirstOutput = True

model = load_model('libraries/network/model/lqm.h5')


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
    test_image = cv2.resize(test_image.copy(), (64, 64), interpolation=cv2.INTER_AREA)
    t = test_image.copy()
    cv2.resize(test_image, (64, 64))
    test_image = (image.img_to_array(test_image)) / 255
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    np.reshape(result, 36)
    high = np.amax(test_image)
    low = np.amin(test_image)
    if high != low:
        maxval = np.amax(result)
        index = np.where(result == maxval)
        arr_out.append(arr_result[index[1][0]])
        if createOutputImage is not False:
            global outputImage
            outputImage = 'output.png'
            return arr_result[index[1][0]]


def predict(input_img, boxColor, boxWidth, createOutput=False):
    im = input_img.copy()
    img = np.array(im)

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


cv2.waitKey()
cv2.destroyAllWindows()
