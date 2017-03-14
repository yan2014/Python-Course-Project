import cv2
import numpy as np
import scipy as sp


def normalizeImage(src_array):
    minarray=np.amin(src_array)
    maxarray=np.amax(src_array)

    for i in range (0,src_array.shape[0]):
        for j in range(0,src_array.shape[1]):
            src_array[i][j]=(src_array[i][j]-minarray)*float(255)/(maxarray-minarray)
    return np.uint8(src_array)

def gradientX(image):
    i=0
    j=0
    imagegradientx=np.empty((image.shape[0],image.shape[1]-1),np.int64)

    for i in range (0,imagegradientx.shape[0]):
        for j in range(0,imagegradientx.shape[1]):
            imagegradientx[i][j]= float(image[i][j+1])-float(image[i][j])
    return imagegradientx


def gradientY(image):
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


def crossCorrelation2D(image, kernel):
    k=kernel.shape[0]
    N=k//2
    image2=np.zeros([image.shape[0]-k+1,image.shape[1]-k+1])
    for i in range(0,image.shape[0]-k+1):
        for j in range(0, image.shape[1]-k+1):
            for m in range(-N,N+1):
                for n in range(-N, N+1):
                    image2[i,j]+=image[i+N+m,j+N+n]*kernel[N+m,N+n]
    return np.float64(image2)


def pyFilter2D(image, kernel):
    padded_image = padReflectBorder(image, kernel.shape[0] // 2)
    filtered_image = crossCorrelation2D(padded_image, kernel)
    return filtered_image
