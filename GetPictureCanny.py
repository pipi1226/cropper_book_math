#!/usr/bin/env python
# --------------------------------------------------------
# ShoeContourDetect
# --------------------------------------------------------

import os, sys, cv2

from matplotlib import pyplot as plt
import numpy as np

import tools as tl

# warpAffineShoe
# wholecard
#img = cv2.imread('warpAffineShoe.jpg', 1)
#img = cv2.imread('test.JPG', 1)
# 2016-09-05 162336
#img = cv2.imread('TT.JPG', 1)
#img = cv2.imread('test.jpg', 1)

def getBlurParam(lowBlur):
    blurvalue = lowBlur

    if blurvalue %2 == 0:
        blurvalue += 1

    imgCpy = cv2.GaussianBlur(img, (blurvalue, blurvalue), 0)
    cv2.imshow('canny demo', imgCpy)
    return imgCpy

# get shoe length
def getCannyParam(lowThreshold):

    #imgCpy = imgShoe.copy()
    #imgCanny, imgCannyDil = tl.getCanny(imgCpy, 5, 7, 1, 13, 13, 7, 7, False)
    # lowThreshold, lowThreshold * ratio, apertureSize = kernel_size
    #imgCanny, imgCannyDil = tl.getCanny(img, lowThreshold, ratio, 1, 13, 13, 7, 7, False)

    imgCanny, imgCannyDil = tl.getCanny(imgCpy, lowThreshold, ratio, lowmulti, lowdilate, lowdilate, lowerode, lowerode, False)

    cv2.imshow('canny demo', imgCannyDil)
    return None

def getShoeLength(imgShoe):
    imgCpy = imgShoe.copy()
    imgCpy = cv2.GaussianBlur(imgCpy, (35, 35), 0)
    imgCanny, imgCannyDil = tl.getCanny(imgCpy, 5, 7, 1, 9, 9, 5, 5, False)
    cv2.imwrite('shoelength.jpg', imgCannyDil)
    return None

lowThreshold = 0
max_lowThreshold = 1000
ratio = 3
kernel_size = 3

lowerode = 1
higherode = 100

lowdilate = 1
highdilate = 100

lowmulti = 1
highmulti = 10

lowBlur = 0
highBlur = 55

def nothing(x):
    pass

if __name__ == "__main__":

    cnt = 0
    listpic = []
    version = cv2.__version__
    print "shell name :", sys.argv[0]
    for i in range(1, len(sys.argv)):
        print "parameter: ", i, sys.argv[i]
    strFile = sys.argv[1]
    img = cv2.imread(strFile, 1)
    
    cv2.namedWindow('canny demo', cv2.WINDOW_NORMAL)
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    imgCpy = cv2.GaussianBlur(img, (35, 35), 0)
    #imgCpy = img.copy()
    cv2.imshow('canny demo', imgCpy)

    '''
    cv2.createTrackbar('Min Blur', 'canny demo', lowBlur, highBlur, getBlurParam)
    cv2.createTrackbar('Min threshold', 'canny demo', lowThreshold, max_lowThreshold, getCannyParam)
    cv2.createTrackbar('ratio', 'canny demo', 0, ratio * 4, getCannyParam)

    cv2.createTrackbar('Min dilate', 'canny demo', lowdilate, highdilate, getCannyParam)
    cv2.createTrackbar('Min erode', 'canny demo', lowerode, highdilate, getCannyParam)
    cv2.createTrackbar('Min multiple', 'canny demo', lowmulti, highmulti, getCannyParam)
    '''

    cv2.createTrackbar('Min Blur', 'canny demo', lowBlur, highBlur, nothing)
    cv2.createTrackbar('Min threshold', 'canny demo', lowThreshold, max_lowThreshold, nothing)
    cv2.createTrackbar('ratio', 'canny demo', 0, ratio * 4, nothing)

    cv2.createTrackbar('Min dilate', 'canny demo', lowdilate, highdilate, nothing)
    cv2.createTrackbar('Min erode', 'canny demo', lowerode, highdilate, nothing)
    cv2.createTrackbar('Min multiple', 'canny demo', lowmulti, highmulti, nothing)

    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'canny demo', 1, 1, nothing)

    while (1):
        cv2.imshow('canny demo', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        blurselect = cv2.getTrackbarPos('Min Blur', 'canny demo')
        thresholdselect = cv2.getTrackbarPos('Min threshold', 'canny demo')
        ratioselect = cv2.getTrackbarPos('ratio', 'canny demo')
        dilateselect = cv2.getTrackbarPos('Min dilate', 'canny demo')
        erodeselect = cv2.getTrackbarPos('Min erode', 'canny demo')
        multiselect = cv2.getTrackbarPos('Min multiple', 'canny demo')

        switchselect = cv2.getTrackbarPos(switch, 'canny demo')
        if switchselect == 0:
            #img[:] = 0
            imgCpy = img.copy()
            cv2.imshow('result', imgCpy)
        else:
            imgCpy = img.copy()
            if blurselect % 2 == 0:
                blurselect += 1

            imgCpy = cv2.GaussianBlur(img, (blurselect, blurselect), 0)

            imgCanny, imgCannyDil = tl.getCanny(imgCpy, thresholdselect, ratioselect, multiselect, dilateselect, dilateselect, erodeselect,
                                                erodeselect, False)

            cv2.imshow('result', imgCannyDil)


    cv2.destroyAllWindows()


