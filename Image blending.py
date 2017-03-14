import numpy as np
import scipy as sp
import scipy.signal
import cv2


def generatingKernel(a):
    kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
    return np.outer(kernel, kernel)


def reduce_layer(image, kernel=generatingKernel(0.4)):   
    image=np.float64(image)
    image = cv2.copyMakeBorder(image,5,5,5,5,cv2.BORDER_REFLECT)
    kernel=np.float64(kernel)
    image= scipy.signal.convolve2d(image, kernel,'same')
    image=image[5:-5, 5:-5]
    image=image[::2, ::2]
    return np.float64(image)

def expand_layer(image, kernel=generatingKernel(0.4)):
    (width, height)=image.shape
    newimage=np.zeros([width*2,height*2])
    newimage[::2, ::2]=image
    newimage= cv2.copyMakeBorder(newimage,5,5,5,5,cv2.BORDER_REFLECT)
    newimage=scipy.signal.convolve2d(newimage, kernel, 'same')*4
    newimage=newimage[5:-5, 5:-5]
    return np.float64(newimage)

def gaussPyramid(image, levels):
    image=np.float64(image)
    pyramidlevel=[image]
    for i in range (levels):
        pyramidlevel.append (np.float64(reduce_layer(pyramidlevel[i])))
    return pyramidlevel

def laplPyramid(gaussPyr):
    laplacianlevel=[]
    for i in range(len(gaussPyr)-1):
        (width,height)=gaussPyr[i].shape
        laplacianlevel.append(gaussPyr[i]-expand_layer(gaussPyr[i+1])[:width,:height])
    laplacianlevel.append(gaussPyr[len(gaussPyr)-1])
    return laplacianlevel

def blend(laplPyrWhite, laplPyrBlack, gaussPyrMask):
    blendlevel=[]
    for i in range(len(laplPyrBlack)):
        blendlevel.append(laplPyrBlack[i]*(1-gaussPyrMask[i])+laplPyrWhite[i]*gaussPyrMask[i])
    return blendlevel

def collapse(pyramid):
    newimage=pyramid[len(pyramid)-1]
    for i in range(len(pyramid)-2,-1,-1):
        (width,height)=pyramid[i].shape
        newimage=expand_layer(newimage)[:width,:height]+pyramid[i]
    return newimage
