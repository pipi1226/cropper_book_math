#!/usr/bin/env python
# --------------------------------------------------------
# FootDetect
# 2016-08-25
# a4 210mm x 297mm
# --------------------------------------------------------

import os, sys, cv2
from matplotlib import pyplot as plt
import numpy as np

import math

# add a black margin
def addBlackMargin(img):
    h, w, s = img.shape
    print 'imgsize = ', w, h, s, 'dtype=', img.dtype
    # start add margin
    size = h+2, w+2, s
    m = np.zeros(size, dtype=np.uint8)
    m[1: h, 1: w] = img[0: h-1, 0: w-1]
    #print 'img[1,1] = ', img[0,0]
    #cv2.imwrite('add.jpg', m)
    # end add margin
    return m

# inverse image
def imgInvert(im_name):
    img = cv2.imread(im_name, 1);
    img = (255 - img)
    # cv2.imwrite('inver.jpg', img)
    return img

# balance hist
def balanceHist(image):
    lut = np.zeros(256, dtype=image.dtype)
    # it is a 1D histogram
    hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])

    minBinNo, maxBinNo = 0, 255

    for binNo, binValue in enumerate(hist):
        if binValue != 0:
            minBinNo = binNo
            break

    for binNo, binValue in enumerate(reversed(hist)):
        if binValue != 0:
            maxBinNo = 255 - binNo
            break
    print minBinNo, maxBinNo

    for i, v in enumerate(lut):
        print i
        if i < minBinNo:
            lut[i] = 0
        elif i > maxBinNo:
            lut[i] = 255
        else:
            lut[i] = int(255.0 * (i - minBinNo) / (maxBinNo - minBinNo) + 0.5)
    return lut


# get white object

def detectWhiteHSV(imgAdd, inx, iselect):
    #img = cv2.imread(im_name, 1)

    # img = cv2.blur(img, (5,5))
    img = cv2.GaussianBlur(imgAdd, (5, 5), 0)

    img2hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img2hsv)

    # img2hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.equalizeHist()
    channels = np.zeros(img.shape, np.uint8)
    imgNhsv = img2hsv.copy()

    cv2.equalizeHist(v, v)
    imgNhsv = cv2.merge((h, s, v))

    lower_white = np.array([0, 0, 46])
    upper_white = np.array([180, 46, 255])

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(imgNhsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    # res = cv2.bitwise_and(h, h, mask= mask)
    res = cv2.bitwise_and(img, img, mask=mask)

    if inx == iselect:
        img_rgb = np.zeros(img.shape, img.dtype)
        # listpic.append(h)
        # listpic.append(res)
        #cv2.imwrite('whitegray.jpg', imgGray)
        #cv2.imwrite('whitethresh.jpg', res)
    return res


# find corner
def findCorner(conArray, w, h):
    corner = []

    lu = [0, 0]
    lw = [0, 0]
    #[conArray[0][0], conArray[0][1]]
    ru = [0, 0]
    rw = [0, 0]

    minW1 = [w, h]
    minW2 = [w, h]
    minH1 = [w, h]
    minH2 = [w, h]

    lmin1 = 0
    lmin2 = 0
    rmin1 = 0
    rmin2 = 0

    for inx, ar in enumerate(conArray):
        # left up
        if minW1 > ar[0]:
            minW1 = ar[0]
            lmin1 = inx
            if minW2 > minW1:
                minW2 = minW1
                lmin2 = inx




    corner.append(lu)
    corner.append(ru)
    corner.append(rw)
    corner.append(lw)
    print 'corner = ',corner
    return corner

def getWarpAffine(img, ptsPers1, ptsPers2, heiResize, widResize):
    warAffMat = cv2.getAffineTransform(ptsPers1, ptsPers2)
    #print 'getWarpAffine(), warAffMat = ', warAffMat
    warpDst = cv2.warpAffine(img, warAffMat, (heiResize, widResize))
    return warpDst


def getWarpPerspect(img, ptsPers1, ptsPers2, widResize, heiResize):
    warPersMat = cv2.getPerspectiveTransform(ptsPers1, ptsPers2)
    warpPersDst = cv2.warpPerspective(img, warPersMat, (widResize, heiResize))
    return warPersMat

def findLargestContour(contours, h, w):
    size = w * h
    max = cv2.contourArea(contours[0])
    inxMax = 0
    for cnt, cons in enumerate(contours):
        area = cv2.contourArea(cons)
        if max < area and area < size:
            max = area
            inxMax = cnt
            # x,y,w,h = cv2.boundingRect(cnt)
            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    print 'max=', max, 'inxMax = ', inxMax, 'maxpoints = ', len(contours[inxMax])
    return inxMax, contours[inxMax]


def getSubMaskImg(img, ptArray, version):
    # get the rectangle sub picture
    size = img.shape
    imgBoxMask = np.full(img.shape, 0, dtype=np.uint8)
    cv2.drawContours(imgBoxMask, [ptArray], 0, (255, 255, 255), -1)
    imgNoBoxMask = cv2.bitwise_not(imgBoxMask)
    imgRes = img.copy()
    imgRes = cv2.bitwise_and(img, imgBoxMask, imgNoBoxMask)
    cv2.imwrite('result.jpg', imgRes)
    return imgRes


def findLongestContour(contours, h, w):
    max = cv2.arcLength(contours[0], False)
    inxMax = 0
    for cnt, cons in enumerate(contours):
        length = cv2.arcLength(cons, False)
        if max < length:
            max = length
            inxMax = cnt
    print 'max=', max, 'inxMax = ', inxMax, 'maxpoints = ', len(contours[inxMax])
    return inxMax, contours[inxMax]


def drawContours(str, img, contour):
    imgCpy = img.copy()
    cv2.drawContours(imgCpy, contour, -1, (0,0,255), 2)
    cv2.imwrite(str, imgCpy)
    return

# compute distance
def computeDist(x1, y1, x2, y2):
    meter = 0.0
    deltax = abs(x1-x2)
    deltay = abs(y1-y2)
    meter = math.sqrt(deltax*deltax + deltay*deltay)
    return meter


# sort box rect coordinates
def sortBoxPoints(box):
    corner = sorted(box, key=lambda box: box[0])
    #print 'before arrange coner = ', corner

    #if corner[0][1] > corner[1][1] and corner[0][0] >= corner[1][0]:
    if corner[0][1] > corner[1][1]:
        #print 'change left up and left down point'
        tmp = corner[0]
        corner[0] = corner[1]
        corner[1] = tmp

    #if corner[2][1] > corner[3][1] and corner[2][0] >= corner[3][0]:
    if corner[2][1] > corner[3][1]:
        #print 'change right up and right down point'
        tmp = corner[2]
        corner[2] = corner[3]
        corner[3] = tmp

    # corner = sorted(corner, key = lambda corner:corner[1])
    #print 'sorted corner =', corner
    return corner

# compute rectangle area of a contour
def computeRectArea(contour, version):
    area = -1.0
    rect = cv2.minAreaRect(contour)
    #print 'rect = ', rect
    if version == '3.1.0':
        box = cv2.boxPoints(rect)
    else:
        box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)

    corner = sortBoxPoints(box)

    wid = computeDist(corner[0][0], corner[0][1], corner[2][0], corner[2][1])
    hei = computeDist(corner[0][0], corner[0][1], corner[1][0], corner[1][1])
    area = wid * hei
    return area, corner, wid, hei

def getCanny(img, edgeTh, multiple, iterCany=1, dilateWid=5, dilateHei=5, erodeWid=3, erodeHei=3, bSub = True, bColr = True):

    if bColr == True:
        imgR, imgG, imgB = cv2.split(img)

        # imgCanny = cv2.Canny(imgGray, edgeTh, edgeTh * 7, 1)
        # cv2.imwrite('beforecannycard.jpg', imgCanny)

        imgRCanny = cv2.Canny(imgR, edgeTh, edgeTh * multiple, iterCany)
        imgBCanny = cv2.Canny(imgB, edgeTh, edgeTh * multiple, iterCany)
        imgGCanny = cv2.Canny(imgG, edgeTh, edgeTh * multiple, iterCany)
        imgCanny = imgRCanny + imgBCanny
        imgCanny = imgCanny + imgGCanny
        #cv2.imwrite('beforecannycard.jpg', imgCanny)

        # dilate canny picture

        kernels_dil = np.ones((dilateWid, dilateHei), np.uint8)
        imgCannyDil = cv2.dilate(imgCanny, kernels_dil, iterations=1)

        kernels_ero = np.ones((erodeWid, erodeHei), np.uint8)
        imgCannyDil = cv2.erode(imgCannyDil, kernels_ero, iterations=1)

        if bSub == True:
            _, imgThresh = cv2.threshold(imgCannyDil, 0, 255, cv2.THRESH_BINARY)
            imgCannyDil = (255 - imgThresh)
    else:

        imgCanny = cv2.Canny(img, edgeTh, edgeTh * multiple, iterCany)

        #cv2.imwrite('beforecannycard.jpg', imgCanny)

        # dilate canny picture

        kernels_dil = np.ones((dilateWid, dilateHei), np.uint8)
        imgCannyDil = cv2.dilate(imgCanny, kernels_dil, iterations=1)

        kernels_ero = np.ones((erodeWid, erodeHei), np.uint8)
        imgCannyDil = cv2.erode(imgCannyDil, kernels_ero, iterations=1)

        if bSub == True:
            _, imgThresh = cv2.threshold(imgCannyDil, 0, 255, cv2.THRESH_BINARY)
            imgCannyDil = (255 - imgThresh)

    #cv2.imwrite('cannydil.jpg', imgCannyDil)

    return imgCanny, imgCannyDil
