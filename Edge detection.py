# ASSIGNMENT 4
# Shiyan Jiang

""" Assignment 4 - Detecting Gradients / Edges

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

NOTE ABOUT RESTRICTED FUNCTIONS:
--------------------------------
Some of the functions in this assignment prohibit the use of certain library
functions. The docstring notes for each function will refer to these lists in
cases wehre they are disallowed.

Operator methods:
    numpy.sum, numpy.add, numpy.subtract, numpy.multiply, numpy.divide,
    cv2.sum, cv2.add, cv2.addWeighted, cv2.multiply, cv2.divide, cv2.subtract,
    cv2.scaleAdd

Convolution functions:
    cv2.filter2D, cv2.matchTemplate, numpy.fft.fft, numpy.fft.fft2,
    numpy.fft.fftn, scipy.fft, scipy.fftpack.fft, scipy.fftpack.fft2,
    scipy.fftpack.fftn
"""
import cv2
import numpy as np
import scipy as sp


def normalizeImage(src_array):
    """Shift and scale the range of values in src_array to fit in the interval
    [0...255]

    This function should shift the range of the input array so that the minimum
    value is equal to 0 and apply a linear scaling to the values in the input
    array such that the maximum value of the input maps to 255.

    The result should be equivalent to the library call:

        cv2.normalize(src_array, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    (Notice that this means that the output array should have the same value
    type as the input array.)

    NOTE: You MAY NOT use any calls to library functions from openCV, scipy, or
          numpy that perform this function directly, nor may you use any of
          the operator methods listed in the note at the top. You MAY use numpy
          operator broadcasting and/or "advanced" indexing techniques.

    Parameters
    ----------
    src_array : numpy.ndarray
        An input array to be normalized.

    Returns
    -------
    numpy.ndarray(dtype=np.uint8)
        The input array after shifting and scaling the value range to fit in
        the interval [0...255]
    """

    minarray=np.amin(src_array)
    maxarray=np.amax(src_array)

    for i in range (0,src_array.shape[0]):
        for j in range(0,src_array.shape[1]):
            src_array[i][j]=(src_array[i][j]-minarray)*float(255)/(maxarray-minarray)
    return np.uint8(src_array)
    raise NotImplementedError

def gradientX(image):
    """Compute the discrete gradient of an image in the X direction.

    NOTE: See lectures 02-06 (Differentiating an image in X and Y) for a good
          explanation of how to perform this operation.

    The X direction means that you are subtracting columns:

        F(x, y) = F(x+1, y) - F(x, y)

    NOTE: Array coordinates are given in (row, column) order, which is the
          opposite of the (x, y) convention used for Euclidean coordinates

    NOTE: You MAY NOT use any calls to library functions from openCV, scipy, or
          numpy that perform this function, nor may you use any of the operator
          methods listed in the note at the top. You MAY use numpy operator
          broadcasting and/or "advanced" indexing techniques.

    Parameters
    ----------
    image : numpy.ndarray(dtype=np.uint8)
        A grayscale image represented in a numpy array.

    Returns
    -------
    numpy.ndarray(dtype=np.int64)
        The image gradient in the X direction. The shape of the output array
        should have a width that is one column less than the original since
        no calculation can be done once the last column is reached.
    """

    i=0
    j=0
    imagegradientx=np.empty((image.shape[0],image.shape[1]-1),np.int64)

    for i in range (0,imagegradientx.shape[0]):
        for j in range(0,imagegradientx.shape[1]):
            imagegradientx[i][j]= float(image[i][j+1])-float(image[i][j])
    return imagegradientx
    raise NotImplementedError


def gradientY(image):
    """Compute the discrete gradient of an image in the Y direction.

    NOTE: See lectures 02-06 (Differentiating an image in X and Y) for a good
          explanation of how to perform this operation.

    The Y direction means that you are subtracting columns:

        F(x, y) = F(x, y+1) - F(x, y)

    NOTE: Array coordinates are given in (row, column) order, which is the
          opposite of the (x, y) convention used for Euclidean coordinates

    NOTE: You MAY NOT use any calls to library functions from openCV, scipy, or
          numpy that perform this function, nor may you use any of the operator
          methods listed in the note at the top. You MAY use numpy operator
          broadcasting and/or "advanced" indexing techniques.

    Parameters
    ----------
    image : numpy.ndarray(dtype=np.uint8)
        A grayscale image represented in a numpy array.

    Returns
    -------
    numpy.ndarray(dtype=np.int64)
        The image gradient in the Y direction. The shape of the output array
        should have a height that is one row less than the original since
        no calculation can be done once the last row is reached.
    """

    i=0
    j=0
    imagegradienty=np.empty((image.shape[0]-1,image.shape[1]),np.int64)

    for i in range (0,imagegradienty.shape[0]):
        for j in range(0,imagegradienty.shape[1]):
            imagegradienty[i][j]= float(image[i+1][j])-float(image[i][j])
    return imagegradienty
    raise NotImplementedError


def padReflectBorder(image, N):
    """This function pads the borders of the input image by reflecting the
    image across the boundaries.

    N is the number of rows or columns that should be added at each border;
    i.e., the output size should have 2N more rows and 2N more columns than
    the input image.

    The values in the input image should be copied to fill the middle of the
    larger array, and the borders should be filled by reflecting the array
    contents as described in the documentation for cv2.copyMakeBorder().

    This function should be equivalent to the library call:

        cv2.copyMakeBorder(image, N, N, N, N, borderType=cv2.BORDER_REFLECT_101)

    Note: BORDER_REFLECT_101 means that the values in the image array are
          reflected across the border. Ex.   gfedcb|abcdefgh|gfedcba

    NOTE: You MAY NOT use any calls to numpy or opencv library functions, but
          you MAY use array broadcasting and "advanced" numpy indexing
          techniques for this function.

    Parameters
    ----------
    image : numpy.ndarray(dtype=np.uint8)
        A grayscale image represented in a numpy array.

    N : int
        An integer strictly greater than zero and less than the smallest
        dimension of the input image representing the number of padding pixels
        to add at each border.

    Returns
    -------
    numpy.ndarray(dtype=np.uint8)
        A copy of the input array with 2N additional rows and columns filled
        with the values of the input image reflected over the borders.
    """

    image3=np.empty((image.shape[0]+2*N,image.shape[1]+2*N),np.uint8)

    for i in range (0,image.shape[0]+2*N):
        for j in range(0,image.shape[1]+2*N):
                if(N>=i and N>=j):
                        image3[i][j]=image[N-i][N-j]
                elif(N<=i and i< image.shape[0]+N and N<=j and j<image.shape[1]+N):
                        image3[i][j]=image[i-N][j-N]
                elif(N>=i and N<=j and j<image.shape[1]+N):
                        image3[i][j]=image[N-i][j-N]
                elif(N<=i and i<image.shape[0]+N and N>=j):
                        image3[i][j]=image[i-N][N-j]
                elif(N<=i and i<image.shape[0]+N and j>= image.shape[1]+N):
                        image3[i][j]=image[i-N][2*image.shape[1]+N-2-j]
                elif(N>=i and j>= image.shape[1]+N):
                        image3[i][j]=image[N-i][2*image.shape[1]+N-2-j]
                elif(i>=image.shape[0]+N and N>=j):
                        image3[i][j]=image[2*image.shape[0]+N-2-i][N-j]
                elif(i>=image.shape[0]+N and N<=j and j<image.shape[1]+N):
                        image3[i][j]=image[2*image.shape[0]+N-2-i][j-N]
                elif(i>=image.shape[0]+N and j>= image.shape[1]+N):
                        image3[i][j]=image[2*image.shape[0]+N-2-i][2*image.shape[1]+N-2-j]
    return image3
    raise NotImplementedError


def crossCorrelation2D(image, kernel):
    """This function uses native Python code & loops to compute and return the
    valid region of the cross correlation of an input kernel applied to each
    pixel of the input array.

    NOTE: Lectures 2-05, 2-06, and 2-07 address this concept.

    Recall that for an image F and kernel h, cross correlation is defined as:

        G(i,j) = sum_u=-k..k sum_v=-k..k h[u,v] F[i+u,j+v]

    For N = kernel.shape[0] // 2, this function should be equivalent to:

        cv2.filter2D(image, cv2.CV_64F, kernel)[N:-N, N:-N]

    See http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#filter2d
    for details.

    Your code must operate on each pixel of the image and kernel individually
    for each step of the computation. (We know this is inefficient, but we want
    to make sure that you understand what is really happening within the more
    efficient library functions that are available.)

    NOTE: You MAY NOT use any numpy, scipy, or opencv library functions,
          broadcasting rules, or "advanced" numpy indexing techniques, nor may
          you use the operator functions or convolution functions listed in the
          note at the top. You MUST manually loop through the image at each
          pixel. (Yes, we know this is slow and inefficient.)

    NOTE: You MAY assume that kernel will always be a square array with an odd
          number of elements.

    Parameters
    ----------
    image : numpy.ndarray(dtype=np.uint8)
        A grayscale image represented in a numpy array.

    kernel : numpy.ndarray
        A kernel represented in a numpy array of size (k, k) where k is an odd
        number strictly greater than zero.

    Returns
    -------
    output : numpy.ndarray(dtype=np.float64)
        The output image. The size of the output array should be smaller than
        the original image size by k-1 rows and k-1 columns, where k is the
        size of the kernel.
    """

    k=kernel.shape[0]
    N=k//2
    image2=np.zeros([image.shape[0]-k+1,image.shape[1]-k+1])
    for i in range(0,image.shape[0]-k+1):
        for j in range(0, image.shape[1]-k+1):
            for m in range(-N,N+1):
                for n in range(-N, N+1):
                    image2[i,j]+=image[i+N+m,j+N+n]*kernel[N+m,N+n]
    return np.float64(image2)
    raise NotImplementedError


def pyFilter2D(image, kernel):
    """This function applies the input kernel to the image by performing 2D
    cross correlation on each pixel of the input image.

    NOTE: Lectures 2-05, 2-06, and 2-07 address this concept.

    When padReflectBorder and crossCorrelation are implemented properly, this
    function is equivalent to the library call:

        cv2.filter2D(image, cv2.CV_16S, kernel, achor=(-1,-1), delta=0,
                     borderType=cv2.BORDER_REFLECT_101)

    See http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#filter2d
    for details.

    NOTE: This function is not graded in the assignment because it is given to
          you, but you may find it helpful for producing output for your
          report. Separating the functions for padding and cross correlation
          allows the autograder to test them independently.

    Parameters
    ----------
    image : numpy.ndarray(dtype=np.uint8)
        A grayscale image represented in a numpy array.

    kernel : numpy.ndarray
        A kernel represented in a numpy array of size (k, k) where k is an odd
        number strictly greater than zero.

    Returns
    -------
    numpy.ndarray
        An image computed by padding the input image border and then performing
        cross correlation with the input kernel.
    """
    # DO NOT CHANGE THE CODE IN THIS FUNCTION
    padded_image = padReflectBorder(image, kernel.shape[0] // 2)
    filtered_image = crossCorrelation2D(padded_image, kernel)
    return filtered_image
