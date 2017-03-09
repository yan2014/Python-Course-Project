# ASSIGNMENT 2
# Shiyan Jiang
# GTID

""" Assignment 2 - Basic Image I/O & Simple Image Processing

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but these functions should NOT save the image to disk.

    2. DO NOT import any other libraries aside from those that we provide.
    You should be able to complete the assignment with the given libraries
    (and in many cases without them).

    3. DO NOT change the format of this file. You may NOT change function
    type signatures (not even named parameters with defaults). You may add
    additional code to this file at your discretion, however it is your
    responsibility to ensure that the autograder accepts your submission.

    4. This file has only been tested in the course virtual environment.
    You are responsible for ensuring that your code executes properly in the
    virtual machine environment, and that any changes you make outside the
    areas annotated for student code do not impact your performance on the
    autograder system.
"""
import cv2
import numpy as np


def numberOfPixels(image):
    """This function returns the number of pixels in a grayscale image.

    Note: A grayscale image has one channel as covered in the lectures. You
    DO NOT need to handle color images.

    You MAY use any/all functions to obtain the number of pixels in the image.

    Parameters
    ----------
    image : numpy.ndarray(dtype=np.uint8)
        A grayscale image represented in a numpy array.

    Returns
    -------
    int
        The number of pixels in an image.  (Note that this return type may
        require type conversion or casting.)
    """
    return int(image.size)
    raise NotImplementedError


def averagePixel(image):
    """This function returns the average pixel intensity of a grayscale image.

    In order to calculate the average pixel intensity, take the sum of all
    pixel intensities (the value of each pixel) and divide by the total number
    of pixels.

    You MAY NOT use numpy.mean, numpy.average, cv2.mean, or any other library
    function that automatically computes the average. All other functions
    are allowed.

    Parameters
    ----------
    image : numpy.ndarray(dtype=np.uint8)
        A grayscale image represented in a numpy array.

    Returns
    -------
    int
        The average pixel in the image.  (Note that this return type may
        require type conversion or casting.)
    """

    sum=0
    i=0
    j=0
    for i in range (0,image.shape[0]):
        for j in range(0,image.shape[1]):
            sum+=image[i][j]

    return int (sum/image.size)
    raise NotImplementedError


def convertToBlackAndWhite(image):
    """This function converts a grayscale image to black and white by
    thresholding on the middle pixel intensity value for 8-bit monochrome
    images.

    To convert the image, iterate through every pixel in the image and set each
    pixel strictly greater than 128 to 255, otherwise set the pixel to 0. You
    are essentially converting the input into a 1-bit image, as we discussed in
    lecture, because the output is a 2-color image.

    You may NOT use any thresholding functions provided by OpenCV to do this.
    All other functions are allowed.

    Parameters
    ----------
    image : numpy.ndarray(dtype=np.uint8)
        A grayscale image represented in a numpy array.

    Returns
    -------
    numpy.ndarray(dtype=np.uint8)
        The black and white image.
    """
    i=0
    j=0
    for i in range (0,image.shape[0]):
        for j in range(0,image.shape[1]):
            if(image[i][j]>128):
                image[i][j]=255
            else:
                image[i][j]=0
    return image
    raise NotImplementedError


def averageTwoImages(image1, image2):
    """This function averages the pixels of the two input images.

    The average image can be calculated by computing the pixel-wise sum of the
    two input images and dividing each pixel by two. (It does not matter
    whether you round or truncate the division.)

    You may use any/all library functions for this function.

    NOTE: You may assume image1 and image2 are the SAME size.

    Parameters
    ---------- 
    image1 : numpy.ndarray(dtype=np.uint8)
        A grayscale image represented in a numpy array.

    image2 : numpy.ndarray(dtype=np.uint8)
        A grayscale image represented in a numpy array.

    Returns
    -------
    numpy.ndarray(dtype=np.uint8)
        The average of image1 and image2
    """
    i=0
    j=0
    image3=np.empty((image1.shape[0],image1.shape[1]),np.uint8)
    for i in range (0,image1.shape[0]):
        for j in range(0,image1.shape[1]):
            image3[i][j]=int(image1[i][j]/2+image2[i][j]/2)
    return image3
    raise NotImplementedError


def flipHorizontal(image):
    """This function flips the input image along the horizontal axis (i.e.,
    flip it across the vertical axis).

    The image can be flipped by switching the first and last column of the
    image, the second and penultimate column, and so on. For example:

    012345        543210
    012345        543210
    012345   ->   543210
    012345        543210
    012345        543210

    You may use any/all library functions for this function.

    Parameters
    ----------
    image : numpy.ndarray(dtyp=np.uint8)
        A grayscale image represented in a numpy array.

    Returns
    -------
    numpy.ndarray(dtype=np.uint8)
        The horizontally flipped image.
    """

    return image[..., ::-1]
    raise NotImplementedError
