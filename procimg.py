import numpy as np
import cv2
from scipy import ndimage
from intro_noise import noisy as n
import math
import matplotlib.pyplot as plt

def getBestShift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)

    rows, cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted

#number is j, label is i

def inp_img(mode,number,noise_op=False):
    #read the image
    if mode is "train":
        gray = cv2.imread("train"+ str(number) + ".png", 0)
    elif mode is "cv":
        gray = cv2.imread("cv" + str(number) + ".png", 0)
    elif mode is "test":
        gray = cv2.imread("test" + str(number) + ".png", 0)
    if noise_op is True:
        gray = n('s&p', gray)
    #rescaling
    gray = cv2.resize(255-gray, (28, 28))
    #white letter with black bckgrnd
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:, 0]) == 0:
        gray = np.delete(gray, 0, 1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:, -1]) == 0:
        gray = np.delete(gray, -1, 1)

    rows, cols = gray.shape
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        #first cols then rows
        gray = cv2.resize(gray, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        #first cols then rows
        gray = cv2.resize(gray, (cols, rows))
    colsPadding = (int(math.ceil((28 - cols) / 2.0)),int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)),int(math.floor((28 - rows) / 2.0)))
    gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')
    shiftx, shifty = getBestShift(gray)
    shifted = shift(gray, shiftx, shifty)
    gray = shifted
    show_img = False
    if show_img is True:
        #print image
        # cv2.imshow('processed digit => '+str(number),gray) NOT WORKING. Have to build OpenCV with cmake
        # cv2.waitKey(0)
        plt.imshow(gray, cmap='gray', interpolation='bicubic')
        #hiding marks on X-Axis and Y-Axis
        plt.xticks([]), plt.yticks([])
        plt.show()
        cv2.imwrite(str(mode)+"_proc_"+str(number)+".png", gray)

    #flatten from 0-255 to 0-1 range
    flatten = gray.flatten() / 255.0
    return flatten
