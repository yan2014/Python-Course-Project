import numpy as np
import scipy as sp
import cv2


def findMatchesBetweenImages(image_1, image_2):
    matches = None       
    image_1_kp = None    
    image_1_desc = None  
    image_2_kp = None    
    image_2_desc = None  

    orb = cv2.ORB()
    image_1_kp, image_1_desc = orb.detectAndCompute(image_1, None)
    image_2_kp, image_2_desc = orb.detectAndCompute(image_2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(image_1_desc, image_2_desc)
    matches = sorted(matches, key = lambda x:x.distance)
    matches = matches[:10]
    return image_1_kp, image_2_kp, matches


def drawMatches(image_1, image_1_keypoints, image_2, image_2_keypoints, matches):
    num_channels = 1
    if len(image_1.shape) == 3:
        num_channels = image_1.shape[2]
    margin = 10
    joined_image = np.zeros((max(image_1.shape[0], image_2.shape[0]),
                            image_1.shape[1] + image_2.shape[1] + margin,
                            3))
    if num_channels == 1:
        for channel_idx in range(3):
            joined_image[:image_1.shape[0],
                         :image_1.shape[1],
                         channel_idx] = image_1
            joined_image[:image_2.shape[0],
                         image_1.shape[1] + margin:,
                         channel_idx] = image_2
    else:
        joined_image[:image_1.shape[0], :image_1.shape[1]] = image_1
        joined_image[:image_2.shape[0], image_1.shape[1] + margin:] = image_2

    for match in matches:
        image_1_point = (int(image_1_keypoints[match.queryIdx].pt[0]),
                         int(image_1_keypoints[match.queryIdx].pt[1]))
        image_2_point = (int(image_2_keypoints[match.trainIdx].pt[0] +
                             image_1.shape[1] + margin),
                         int(image_2_keypoints[match.trainIdx].pt[1]))

        rgb = (np.random.rand(3) * 255).astype(np.int)
        cv2.circle(joined_image, image_1_point, 5, rgb, thickness=-1)
        cv2.circle(joined_image, image_2_point, 5, rgb, thickness=-1)
        cv2.line(joined_image, image_1_point, image_2_point, rgb, thickness=3)

    return joined_image
