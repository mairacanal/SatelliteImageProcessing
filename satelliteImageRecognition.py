import cv2 as cv
import numpy as np

INPUT_IMAGE = '05-abr-2016.jpg'

def cleanUp(img):

    greyImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(greyImg,(5,5),0)
    ret, mask = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    kernel = np.ones((5,5),np.uint8)
    removedNoiseMask = cv.morphologyEx(greyImg, cv.MORPH_OPEN, kernel)

    newImg = cv.bitwise_and(img, img, mask=removedNoiseMask)
    
    return newImg

def cloud_extracting (img):

    greyImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #ret,th1 = cv.threshold(img,200,255,cv.THRESH_BINARY_INV)
    #ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    #th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
    #th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

    blur = cv.GaussianBlur(greyImg,(5,5),0)
    ret, mask = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    mask_inv = cv.bitwise_not(mask)
    newImg = cv.bitwise_and(img, img, mask=mask_inv)

    return newImg

def ocean_extracting (img): 
    hsvImg = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower_blue = np.array([90,0,0])
    upper_blue = np.array([280, 255, 255])

    mask = cv.inRange(hsvImg, lower_blue, upper_blue)
    mask_inv = cv.bitwise_not(mask)

    newImg = cv.bitwise_and(img, img, mask=mask_inv)

    return newImg

if __name__ == "__main__":
    img = cv.imread(INPUT_IMAGE)
    cv.imshow('Original', cv.resize(img, (496,519), interpolation=cv.INTER_AREA))

    imgWithoutOcean = ocean_extracting(img)
    cv.imshow('Without Ocean', cv.resize(imgWithoutOcean, (496,519), interpolation=cv.INTER_AREA))
    
    imgWithoutClouds = cloud_extracting(imgWithoutOcean)
    cv.imshow('Without Clouds', cv.resize(imgWithoutClouds, (496,519), interpolation=cv.INTER_AREA))

    imgClean = cleanUp(imgWithoutClouds)
    cv.imshow('Cleanup ', cv.resize(imgClean, (496,519), interpolation=cv.INTER_AREA))

    cv.imwrite(INPUT_IMAGE[:-4] + '_Clean.jpg', imgClean)
    
    cv.waitKey(0)
    cv.destroyAllWindows()