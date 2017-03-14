import cv2
import numpy as np


def numberOfPixels(image):
    return int(image.size)


def averagePixel(image):
    sum=0
    i=0
    j=0
    for i in range (0,image.shape[0]):
        for j in range(0,image.shape[1]):
            sum+=image[i][j]

    return int (sum/image.size)


def convertToBlackAndWhite(image):
    i=0
    j=0
    for i in range (0,image.shape[0]):
        for j in range(0,image.shape[1]):
            if(image[i][j]>128):
                image[i][j]=255
            else:
                image[i][j]=0
    return image


def averageTwoImages(image1, image2):
    i=0
    j=0
    image3=np.empty((image1.shape[0],image1.shape[1]),np.uint8)
    for i in range (0,image1.shape[0]):
        for j in range(0,image1.shape[1]):
            image3[i][j]=int(image1[i][j]/2+image2[i][j]/2)
    return image3


def flipHorizontal(image):
    return image[..., ::-1]
