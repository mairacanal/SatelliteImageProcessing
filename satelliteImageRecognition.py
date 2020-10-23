import cv2 as cv
import numpy as np
import os

INPUT_IMAGE = 'new.png'
INPUT_DIR = 'IMGS_SAT/'

def cloud_extracting (img):

    hsvImg = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower_cloud = np.array([0,0,10])
    upper_cloud = np.array([360, 50, 255])

    mask = cv.inRange(hsvImg, lower_cloud, upper_cloud)
    mask_inv = cv.bitwise_not(mask)

    newImg = cv.bitwise_and(img, img, mask=mask_inv)

    return newImg, mask, mask_inv

def ocean_extracting (img): 

    hsvImg = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower_blue = np.array([90,0,0])
    upper_blue = np.array([280, 255, 255])

    mask = cv.inRange(hsvImg, lower_blue, upper_blue)
    mask_inv = cv.bitwise_not(mask)

    newImg = cv.bitwise_and(img, img, mask=mask_inv)

    return newImg

def image_overlap (imgDir):
    base = cv.imread(imgDir + os.listdir(imgDir)[0], 1)

    for file in os.listdir(imgDir):
        if file != os.listdir(imgDir)[0]:
            img = cv.imread(imgDir + file)
            newImg, mask, mask_inv = cloud_extracting(base)

            img_soma = cv.add(img, img, mask=mask)

            bg = cv.bitwise_and(base, base, mask=mask_inv)
            fg = cv.bitwise_and(img_soma, img_soma, mask=mask)

            base = cv.add(bg, fg)

    finalImg = ocean_extracting(base)
    return finalImg

def clustering (img):
    pixels = img.reshape((-1,3))
    pixels = np.float32(pixels)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flag = cv.KMEANS_RANDOM_CENTERS
    k = 5
    ret, label, center = cv.kmeans(pixels, k, None, criteria, 10, flag)    
    
    center = np.uint8(center)
    newImg = center[label.flatten()].reshape((img.shape))

    return newImg

if __name__ == "__main__":
    img = cv.imread(INPUT_IMAGE)
    cv.imshow('Original', cv.resize(img, (496,519), interpolation=cv.INTER_AREA))

    imgWithoutOcean = ocean_extracting(img)
    cv.imshow('Without Ocean', cv.resize(imgWithoutOcean, (496,519), interpolation=cv.INTER_AREA))
    
    imgWithoutClouds, mask, mask_inv = cloud_extracting(imgWithoutOcean)
    cv.imshow('Without Clouds', cv.resize(imgWithoutClouds, (496,519), interpolation=cv.INTER_AREA))

    clusteredImg = clustering(imgWithoutClouds)
    cv.imshow('K-Means Clustering', cv.resize(clusteredImg, (496,519), interpolation=cv.INTER_AREA))

    # finalImg = image_overlap(INPUT_DIR)

    cv.imwrite('Final.jpg', clusteredImg)
    
    cv.waitKey(0)
    cv.destroyAllWindows()