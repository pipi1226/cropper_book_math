# -*- coding: utf-8 -*-
# show cut jpg binary result
import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import math

from inspect import currentframe, getframeinfo

import tools as tl

import glob

frameinfo = getframeinfo(currentframe())

nPageCnt = 0

kThinLineFraction = 20
kMinLineLengthFraction = 4
dpi = 72

def findLargestContour(contours, h, w):
    size = w * h * 1.0
    print 'size = ', type(size)
    maxArea = cv2.contourArea(contours[0])
    print ' maxArea =', type(maxArea)
    inxMax = 0
    for cnt, cons in enumerate(contours):
        area = cv2.contourArea(cons)
        print 'area=', type(area)
        if (maxArea < area) and (area < size):
            maxArea = area
            inxMax = cnt
            # x,y,w,h = cv2.boundingRect(cnt)
            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    print 'max=', maxArea, 'inxMax = ', inxMax, 'maxpoints = ', len(contours[inxMax])
    return inxMax, contours[inxMax]

def tesscomp_origin(img):
    
    max_line_width = dpi/kThinLineFraction
    min_line_len = dpi/kMinLineLengthFraction
    closing_brick = max_line_width / 3
    if(closing_brick %2):
        closing_brick += 1

    h, w= img.shape[:2]

    kernel = np.ones((closing_brick,closing_brick), np.uint8)
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    thopen1 = cv2.erode(th2, kernel, iterations=2)
    thopen = cv2.dilate(thopen1, kernel, iterations=1)

    # use dilate result to get rectangle region. inner rectangle
    contours, hierarchy = cv2.findContours(thopen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    inxMax, maxCont = findLargestContour(contours, h, w)

    x,y,w,h = cv2.boundingRect(maxCont)
    cv2.rectangle(img,(x+10,y+10),(x+w-10,y+h-10),(0,255,0),2)
    # imgOut = cv2.drawContours(img1, contours, inxMax, (0,0,255), 2)
    imgOut = img[y-5:y+5+h, x-5:x+5+w]
    # return bRet, imgOut
    return thopen
    # return thopen

def tesscomp(img):
    
    max_line_width = dpi/kThinLineFraction
    min_line_len = dpi/kMinLineLengthFraction
    closing_brick = max_line_width / 3 * 3
    if(closing_brick %2):
        closing_brick += 1

    h, w= img.shape[:2]

    kernel = np.ones((closing_brick,closing_brick), np.uint8)
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    thopen1 = cv2.erode(th2, kernel, iterations=2)
    thopen = cv2.dilate(thopen1, kernel, iterations=1)

    # use dilate result to get rectangle region. inner rectangle
    contours, hierarchy = cv2.findContours(thopen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    inxMax, maxCont = findLargestContour(contours, h, w)

    x,y,w,h = cv2.boundingRect(maxCont)
    cv2.rectangle(img,(x+10,y+10),(x+w-10,y+h-10),(0,255,0),2)
    # imgOut = cv2.drawContours(img1, contours, inxMax, (0,0,255), 2)
    imgOut = img[y-5:y+5+h, x-5:x+5+w]
    # return bRet, imgOut
    # return img
    return thopen

if __name__ == '__main__':

    global image

    nPageCnt = 0
    filenames = [img for img in glob.glob("cut/*.jpg")]

    # filenames.sort() # ADD THIS LINE
    # print 'size = ', filenames.count

    for imgName in filenames:
        print 'name = ', imgName[8:-4]
        imgName = "0091.JPG"
        imgOrign = cv2.imread(imgName, 0)
        h, w= imgOrign.shape[:2]
        
        # pagename = './crop20200118/'+imgName[8:-4]+'.jpg'
        pagename = './crop20200118/09.jpg'
        dst = tesscomp(imgOrign)
        cv2.imwrite(pagename, dst)

        break
    
    print 'end...'

