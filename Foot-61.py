#!/usr/bin/env python
# --------------------------------------------------------
# FootDetect
# 2016-08-23
# a4 210mm x 297mm
# 2016-08-29
# get the shoe length
# 2016-08-31
# Perspective transformation rectify the rectangle
# 2016-09-01
# optimization
# get white paper region
# waston good
#2016-09-06
# optimize card region
# 2016-09-07
# shoe parameter different from card
# 2016-09-09
# optimization of canny parameter to get card 10cm-100cm
# --------------------------------------------------------

import os, sys, cv2

from matplotlib import pyplot as plt
import numpy as np

import tools as tl

listpic = []

version = ''

# camera pixel parameter
longPic = 3264
shortPic = 2448
fRatio = 3038.3
fwidparameter = 25825.55
fheiparameter = 16415.9349

#fMaxArea = 516.5 * 328.3 * 1.5
fMaxArea = 516.5 * 328.3 * 10
fMinArea = 258.25 * 164.15 * 0.1

# card standard width and height
fCardWid = 8.575
fCardHeight = 5.403

# camera height from land
cameraHeight = -1.0

# card long edge rank ratio
longEdgeKRatio = 0.0
longEdge = 0
shortEdge = 0

EPSILONG = 0.0000001
INFINIT = 999999

# get card width and height
def getCardWidthAndHeight(imgCard):
    wid = -1
    hei = -1

    global version
    global fMinArea, fMaxArea
    img = cv2.pyrMeanShiftFiltering(imgCard, 25, 10)
    #img = cv2.GaussianBlur(imgCard, (5, 5), 0)
    img = imgCard.copy()
    cv2.imwrite('pymidCard.jpg', img)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #imgCanny, imgCannyDil = tl.getCanny(img, 100, 3, 3, 11, 11, 7, 7, True)
    
    himg, wimg, _ = imgCard.shape
    
    if himg < 200 and wimg < 200:
        print 'getCardWidthAndHeight(), card area is smaller than 40000'
        imgCanny, imgCannyDil = tl.getCanny(img, 45, 3, 1, 7, 7, 5, 5, True)
    else:
        print 'getCardWidthAndHeight(), card area is bigger than 40000'
        #imgCanny, imgCannyDil = tl.getCanny(img, 100, 3, 1, 11, 11, 7, 7, True)
	imgCanny, imgCannyDil = tl.getCanny(img, 85, 2, 1, 11, 11, 7, 7, True)

    cv2.imwrite('cardcanny.jpg', imgCannyDil)

    kernels_dil = np.ones((5, 5), np.uint8)
    gradient = cv2.morphologyEx(imgCanny, cv2.MORPH_GRADIENT, kernels_dil)
    newSub = gradient
    cv2.imwrite('gradientcard.jpg', newSub)

    # get card region
    #_, contours, hierarchy = cv2.findContours(imgCannyDil, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    _, contours, hierarchy = cv2.findContours(newSub, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if contours is None:
        print 'getCardWidthAndHeight(), contour is none'
        return -1, -1

    max = -1
    inxMax = -1
    w, h, s = img.shape
    size = w * h

    chooseCnt = 0
    minRatio = 2.0

    smallestArea = longPic * shortPic * 0.001

    corner = []
    bCardRot = 0
    cntMax = 0

    smallestArea = longPic * shortPic * 0.001
    
    area, corner, wid, hei = tl.computeRectArea(contours[0], version)
    print 'before for cycle corner = ', corner
    if wid < hei:
        maxLen = hei
        hei = wid
        wid = maxLen
        bRot = 1
    
    minRatio = 1.0*(wid/hei)

    for cnt, cons in enumerate(contours):
        area, _, wid, hei = tl.computeRectArea(cons, version)
        bRot = 0
        if hei <= 0:
            continue
        if wid < hei:
            maxLen = hei
            hei = wid
            wid = maxLen
            bRot = 1

        ratio = 1.0 * (wid / hei)

        if abs(ratio - 1.57) < abs(minRatio - 1.57) and area > size / 3:
            minRatio = ratio
            chooseCnt = cnt
            bCardRot = bRot
            cntMax = cnt

    # grabcut method to get card region more approximately
    imgBoxMask = np.full(img.shape, 0, dtype=np.uint8)
    imgCardSave = imgCard.copy()
    #cv2.drawContours(imgCardSave, contours, cntMax, (0, 255, 255), -1)
    cv2.drawContours(imgCardSave, contours, cntMax, (0, 255, 255), 2)
    cv2.imwrite('maskcardContour.jpg', imgCardSave)


    area, corner, wid, hei = tl.computeRectArea(contours[cntMax], version)
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (corner[0][0], corner[0][1], corner[3][0], corner[3][1])
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    imgs = img * mask2[:, :, np.newaxis]

    cv2.imwrite('cutcard.jpg', imgs)
    imgsGray = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgsGray, 1, 255, cv2.THRESH_BINARY)
    cv2.imwrite('cutcardBINARY.jpg', thresh)

    #imgsGray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    imgCannyGauss = cv2.Canny(thresh, 3, 9, 1)
    firstContMax = contours[cntMax]

    _, contoursCard, hierarchy = cv2.findContours(imgCannyGauss, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contoursCard) == 0:
        print 'getCardWidthAndHeight(), second contours is none'
        # largest contour dilate out 3 pixel
        area, corner, wid, hei = tl.computeRectArea(firstContMax, version)
        drawCont = firstContMax
    else:
        cntMax = 0
        maxArea, _, _, _ = tl.computeRectArea(contours[0], version)
        for cnt, cons in enumerate(contoursCard):
            area, corner, tmpWid, tmpHei = tl.computeRectArea(cons, version)
            if area > maxArea:
                maxArea = area
                wid = tmpWid
                hei = tmpHei
                cntMax = cnt

                # largest contour dilate out 3 pixel
        area, corner, wid, hei = tl.computeRectArea(contoursCard[cntMax], version)
        drawCont = contoursCard[cntMax]

    print 'getCardWidthAndHeight(), card width = ', wid, 'card height = ', hei

    print 'getCardWidthAndHeight(), card corner = ', corner
    if wid < hei:
        tmp = wid
        wid = hei
        hei = tmp

    longedge = wid
    global cameraHeight
    cameraHeight = (fCardWid) * fRatio / longedge

    print 'getCardWidthAndHeight(), camera height = ', cameraHeight
    # cv2.drawContours(img, drawCont, -1, (0, 0, 255), 2)

    cv2.line(img, (corner[0][0], corner[0][1]), (corner[2][0], corner[2][1]), (0, 255, 255), 2)
    cv2.line(img, (corner[2][0], corner[2][1]), (corner[3][0], corner[3][1]), (0, 255, 255), 2)
    cv2.line(img, (corner[3][0], corner[3][1]), (corner[1][0], corner[1][1]), (0, 255, 255), 2)
    cv2.line(img, (corner[0][0], corner[0][1]), (corner[1][0], corner[1][1]), (0, 255, 255), 2)

    cv2.imwrite('cardcontour.jpg', img)
    return wid, hei


# get card contour, longedge k rank ratio
def getContourCard(img, thresh):

    global longEdgeKRatio
    global longEdge, shortEdge

    if version == '3.1.0':
        #_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    print 'getContourCard(), Contours = ', len(contours)
    max = -1
    inxMax = -1
    w, h, s = img.shape
    size = w * h
    
    chooseCnt = -1
    minRatio = INFINIT
    
    corner = []
    longEdge = 0
    shortEdge = 0
    bCardRot = 0
    bRot = 0
    smallestArea = longPic * shortPic * 0.001
    '''
    area, corner, wid, hei = tl.computeRectArea(contours[0], version)
    print 'before for cycle corner = ', corner
    if wid < hei:
        maxLen = hei
        hei = wid
        wid = maxLen
        bRot = 1
    
    bCardRot = bRot
    minRatio = 1.0*(wid/hei)
    '''
    
    for cnt, cons in enumerate(contours):
        '''
        if cnt == 0:
            continue
        '''
        area, _, wid, hei = tl.computeRectArea(cons, version)
        bRot = 0
        if wid < hei:
            maxLen = hei
            hei = wid
            wid = maxLen
            bRot = 1

        ratio = 1.0 * (wid / hei)
	    
        if abs(ratio - 1.57) < abs(minRatio - 1.57) and area > fMinArea and area < fMaxArea:
        #if abs(ratio - 1.57) < abs(minRatio - 1.57):
            print 'getContourCard(), cnt=', cnt, 'ratio = ', ratio, 'area = ', area, 'minArea = ', fMinArea, 'fMaxArea = ', fMaxArea
            minRatio = ratio
            chooseCnt = cnt
            bCardRot = bRot

    if chooseCnt == -1:
        print 'getContourCard(), couldnot find card contour!!!.....'

    print 'getContourCard(), cnt = ', chooseCnt, 'ratio = ', minRatio
    area, corner, wid, hei = tl.computeRectArea(contours[chooseCnt], version)
    # save card region jpg
    imgTest = img.copy()
    cv2.drawContours(imgTest, contours, chooseCnt, (0, 0, 255), 2)

    strPath = 'wholecard.jpg'
    cv2.imwrite(strPath, imgTest)

    print 'getContourCard(), corner = ', corner

    if len(corner) == 0:
        print 'getContourCard(), corner is none'
        return None, None, None

    corner = tl.sortBoxPoints(corner)
    print 'getContourCard(), after sort corner = ', corner
    longEdge = tl.computeDist(corner[0][0], corner[0][1], corner[1][0], corner[1][1])
    shortEdge = tl.computeDist(corner[0][0], corner[0][1], corner[2][0], corner[2][1])


    if longEdge < shortEdge:
        print 'getContourCard(), longEdge < shortEdge'
        if corner[2][0] == corner[0][0]:
            longEdgeKRatio = INFINIT
        elif corner[2][1] == corner[0][1]:
            longEdgeKRatio = 0.0
        else:
            longEdgeKRatio = (1.0* (corner[2][1] - corner[0][1])) / (1.0 * (corner[2][0] - corner[0][0]))
    else:
        print 'getContourCard(), longEdge >= shortEdge'
        if corner[1][0] == corner[0][0]:
            longEdgeKRatio = INFINIT
        elif corner[1][1] == corner[0][1]:
            longEdgeKRatio = 0.0
        else:
            longEdgeKRatio = (1.0 * (corner[1][1] - corner[0][1])) / (1.0 * (corner[1][0] - corner[0][0]))

    if longEdgeKRatio - INFINIT > EPSILONG and longEdgeKRatio > EPSILONG:
        longEdgeKRatio = INFINIT
    elif longEdgeKRatio < EPSILONG and abs(longEdgeKRatio) - INFINIT > EPSILONG:
        longEdgeKRatio = EPSILONG / 2 - INFINIT

    print 'getContourCard(), longEdgeKRatio = ', longEdgeKRatio
    # get card region
    if bCardRot == 0:
        print 'getContourCard(), do not rotate'
    else:
        print 'getContourCard(), rotate'
        pts1 = np.float32([[corner[2][0], corner[2][1]], [corner[0][0], corner[0][1]], [corner[3][0], corner[3][1]]])
        pts2 = np.float32([[0, 0], [0, (int)(shortEdge)], [(int)(longEdge), 0]])

    pts1 = np.float32([[corner[0][0], corner[0][1]], [corner[2][0], corner[2][1]], [corner[1][0], corner[1][1]]])
    pts2 = np.float32([[0, 0], [0, (int)(shortEdge)], [(int)(longEdge), 0]])

    warpDst = tl.getWarpAffine(img, pts1, pts2, (int)(longEdge), (int)(shortEdge))
    cv2.imwrite('warpAffine.jpg', warpDst)

    _, _ = getCardWidthAndHeight(warpDst)

    return contours[chooseCnt]

# get shoe sub picture
def getShoeLength(imgShoe):

    imgShoeCpy = cv2.GaussianBlur(imgShoe, (35,35), 0)
    imgCanny, imgCannyDil = tl.getCanny(imgShoeCpy, 16, 2, 1, 5, 5, 2, 2, False)
    #imgCanny, imgCannyDil = tl.getCanny(imgShoeCpy, 15, 2, 1, 3, 3, 1, 1, False)

    if version == '3.1.0':
        #_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, contours, hierarchy = cv2.findContours(imgCannyDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(imgCannyDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    print 'getShoeLength(), Contours = ', len(contours)
    max = -1
    inxMax = -1
    w, h, s = img.shape
    size = w * h
    listContour = []
    chooseCnt = 0
    minRatio = 2.0

    smallestArea = longPic * shortPic * 0.001
    largestContour = contours[0]
    largestInx = 0
    maxArea, _, _, _ = tl.computeRectArea(contours[0], version)

    corner = []
    longEdge = 0
    shortEdge = 0
    bCardRot = 0
    for cnt, cons in enumerate(contours):

        area, _, wid, hei = tl.computeRectArea(cons, version)
        if area > maxArea:
            maxArea = area
            largestContour = cons
            largestInx = cnt

    newmask = np.full(imgShoe.shape, 0, dtype=np.uint8)
    cv2.drawContours(newmask, [largestContour], -1, (255, 255, 255), -1)

    kernels_dil = np.ones((3, 3), np.uint8)
    gradient = cv2.morphologyEx(newmask, cv2.MORPH_GRADIENT, kernels_dil)
    newSub = gradient
    cv2.imwrite('gradient.jpg', newSub)

    rect = cv2.minAreaRect(largestContour)

    if version == '3.1.0':
        box = cv2.boxPoints(rect)
    else:
        box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)

    corner = tl.sortBoxPoints(box)
    print 'getShoeLength(), shoe corner=', corner

    cv2.line(newSub, (corner[0][0], corner[0][1]), (corner[2][0], corner[2][1]), (0, 255, 255), 2)
    cv2.line(newSub, (corner[2][0], corner[2][1]), (corner[3][0], corner[3][1]), (0, 255, 255), 2)
    cv2.line(newSub, (corner[3][0], corner[3][1]), (corner[1][0], corner[1][1]), (0, 255, 255), 2)
    cv2.line(newSub, (corner[0][0], corner[0][1]), (corner[1][0], corner[1][1]), (0, 255, 255), 2)
    cv2.imwrite('gradientRect.jpg', newSub)

    wid = tl.computeDist(corner[0][0], corner[0][1], corner[2][0], corner[2][1])
    hei = tl.computeDist(corner[0][0], corner[0][1], corner[1][0], corner[1][1])

    if wid < hei:
        tmp = hei
        hei = wid
        wid = tmp

    print 'getShoeLength(), shoe wid pixel = ', wid, 'height pixel = ', hei
    shoewid = -1
    if cameraHeight > 0:
        shoewid = cameraHeight * wid / 3038.3
    print 'getShoeLength(), shoe wid = ', shoewid

    return None

# get shoe contour
def getContourShoe(img, thresh):
    boxes = []

    global version
    if version == '3.1.0':
        #_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    print 'getContourShoe(), Contours = ', len(contours)

    max = -1
    inxMax = -1
    w, h, s = img.shape
    size = w * h
    chooseCnt = 0

    smallestArea = longPic * shortPic * 0.001
    largestContour = contours[0]
    largestInx = 0
    maxArea, _, _, _ = tl.computeRectArea(contours[0], version)

    corner = []
    for cnt, cons in enumerate(contours):
        area, _, wid, hei = tl.computeRectArea(cons, version)
        if area > maxArea:
            maxArea = area
            largestContour = cons
            largestInx = cnt

    # shoe contour mask
    box = largestContour
    newmask = np.full(img.shape, 0, dtype=np.uint8)
    cv2.drawContours(newmask, [box], 0, (255, 255, 255), -1)
    cv2.imwrite('shoecontour.jpg', newmask)

    # gradient shoe
    kernels_dil = np.ones((3, 3), np.uint8)
    #gradient = cv2.morphologyEx(newmask, cv2.MORPH_GRADIENT, kernels_dil)
    gradient = cv2.morphologyEx(newmask, cv2.MORPH_GRADIENT, kernels_dil)
    newSub = gradient
    cv2.imwrite('shoegradient.jpg', newSub)

    # shoe contour min rect area
    rect = cv2.minAreaRect(largestContour)

    if version == '3.1.0':
        box = cv2.boxPoints(rect)
    else:
        box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)

    corner = tl.sortBoxPoints(box)
    print 'getContourShoe(), shoe corner=', corner

    # get shoe information
    region = img.copy()
    cv2.line(region, (corner[0][0], corner[0][1]), (corner[2][0], corner[2][1]), (0, 0, 255), 2)
    cv2.line(region, (corner[2][0], corner[2][1]), (corner[3][0], corner[3][1]), (0, 0, 255), 2)
    cv2.line(region, (corner[3][0], corner[3][1]), (corner[1][0], corner[1][1]), (0, 0, 255), 2)
    cv2.line(region, (corner[0][0], corner[0][1]), (corner[1][0], corner[1][1]), (0, 0, 255), 2)
    cv2.imwrite('region.jpg', region)

    wid = tl.computeDist(corner[0][0], corner[0][1], corner[2][0], corner[2][1])
    hei = tl.computeDist(corner[0][0], corner[0][1], corner[1][0], corner[1][1])

    pts1 = np.float32([[corner[0][0], corner[0][1]], [corner[1][0], corner[1][1]], [corner[2][0], corner[2][1]]])
    pts2 = np.float32([[0, 0], [0, (int)(hei)], [(int)(wid), 0]])

    # grabcut shoe region
    '''
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (corner[0][0], corner[0][1], corner[3][0], corner[3][1])
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    imgs = img * mask2[:, :, np.newaxis]

    cv2.imwrite('cutshoe.jpg', imgs)
    warpDst = tl.getWarpAffine(imgs, pts1, pts2, (int)(wid), (int)(hei))
    '''

    warpDst = tl.getWarpAffine(img, pts1, pts2, (int)(wid), (int)(hei))
    cv2.imwrite('warpAffineShoe.jpg', warpDst)

    print 'getContourShoe(), shoe picture wid = ', wid, 'hei = ', hei

    getShoeLength(warpDst)

    return

# get shoe length method 2
def getContourShoe2(img, thresh):
    boxes = []

    global version
    if version == '3.1.0':
        # _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    print 'getContourShoe2(), Contours = ', len(contours)

    max = -1
    inxMax = -1
    w, h, s = img.shape
    size = w * h
    chooseCnt = 0

    smallestArea = longPic * shortPic * 0.001
    largestContour = contours[0]
    largestInx = 0
    maxArea, _, _, _ = tl.computeRectArea(contours[0], version)

    corner = []
    for cnt, cons in enumerate(contours):
        area, corner, wid, hei = tl.computeRectArea(cons, version)
        if area > maxArea:
            maxArea = area
            largestContour = cons
            largestInx = cnt

    # shoe contour mask
    box = largestContour
    newmask = np.full(img.shape, 0, dtype=np.uint8)
    cv2.drawContours(newmask, [box], 0, (255, 255, 255), -1)
    cv2.imwrite('shoecontour.jpg', newmask)

    #print 'contour[0] = ', largestContour[0]

    tmp1 = largestContour[0]
    tmp2 = largestContour[1]
    x1 = tmp1[0][0]
    y1 = tmp1[0][1]
    x2 = tmp2[0][0]
    y2 = tmp2[0][1]

    maxDist = tl.computeDist(x1, y1, x2, y2)

    if y1 == y2:
        minRankRatio = 0.0
    elif x1 == x2:
        minRankRatio = INFINIT
    else:
        minRankRatio = (y1-y2) / (x1-x2)
        if minRankRatio - INFINIT > EPSILONG and minRankRatio > EPSILONG:
            minRankRatio = INFINIT
        elif minRankRatio < EPSILONG and abs(minRankRatio) - INFINIT > EPSILONG:
            minRankRatio = EPSILONG / 2 - INFINIT

    print 'getContourShoe2(), longEdgeRankRatio=', longEdgeKRatio
    maxEdge = longEdge
    if longEdge - shortEdge < 0.0001:
        maxEdge = shortEdge

    maxEdge = 1.5 * maxEdge
    for cnt1, cord1 in enumerate(largestContour):
        #print cnt1, cord1[0]
        label1 = cord1[0]
        for cnt2, cord2 in enumerate(largestContour):
            if cnt2 == cnt1:
                continue

            label2 = cord2[0]

            if abs(label1[0] - label2[0]) <= maxEdge and abs(label1[1] - label2[1]) <= maxEdge:
                continue

            tmpDist = 0.0
            tmpRankRatio = 0.0

            if label1[0] == label2[0]:
                tmpDist = abs(label1[1] - label2[1])
                tmpRankRatio = 0.0
            elif label1[1] == label2[1]:
                tmpDist = abs(label1[1] - label2[1])
                tmpRankRatio = INFINIT
            else:
                tmpDist = tl.computeDist(label1[0], label1[1], label2[0], label2[1])
                tmpRankRatio = (label1[1] - label2[1]) / (label1[0] - label2[0])
                if tmpRankRatio - INFINIT > EPSILONG and tmpRankRatio > EPSILONG:
                    tmpRankRatio = INFINIT
                elif tmpRankRatio < EPSILONG and abs(tmpRankRatio) - INFINIT > EPSILONG:
                    tmpRankRatio = EPSILONG / 2 - INFINIT

            if abs(minRankRatio - longEdgeKRatio) >= abs(tmpRankRatio - longEdgeKRatio) and \
                                    tmpDist - maxDist > 0.001:
                maxDist = tmpDist
                minRankRatio = tmpRankRatio
                x1 = label1[0]
                y1 = label1[1]
                x2 = label2[0]
                y2 = label2[1]



    ShoeWid = maxDist - 10
    cv2.circle(img, (x1, y1), 3, (0,0,255), 3)
    cv2.circle(img, (x2, y2), 3, (0, 0, 255), 3)
    cv2.line(img, (x1,y1), (x2, y2), (0,255,255),2)
    cv2.imwrite('circle.jpg', img)
    print 'getContourShoe2(), shoe pixel=', ShoeWid
    if cameraHeight > 0:
        ShoeWid = cameraHeight * ShoeWid / 3038.3
    print 'getContourShoe2(), shoe wid=', ShoeWid, '(cm)'
    return

def getContourShoe3(img):

    global version
    imgBlur = cv2.GaussianBlur(img, (35, 35), 0)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('black.jpg', imgGray)

    #_, thresh = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY)
    thresh = cv2.adaptiveThreshold(imgGray, 50,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
    cv2.imwrite('thresh.jpg', thresh)

    imgCanny, imgCannyDil = tl.getCanny(imgGray, 6, 2, 1, 11, 11, 1, 1, False, False)
    imgCannyDil = (255-imgCannyDil)

    if version == '3.1.0':
        # _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, contours, hierarchy = cv2.findContours(imgCannyDil, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(imgCannyDil, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    h, w = imgGray.shape
    inx, largestContour = tl.findLargestContour(contours, h, w)

    cv2.drawContours(img, largestContour, -1, (0,0,255), 2)

    rect = cv2.minAreaRect(largestContour)

    if version == '3.1.0':
        box = cv2.boxPoints(rect)
    else:
        box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)

    corner = tl.sortBoxPoints(box)
    print 'getShoeLength(), shoe corner=', corner

    cv2.line(img, (corner[0][0], corner[0][1]), (corner[2][0], corner[2][1]), (0, 255, 255), 2)
    cv2.line(img, (corner[2][0], corner[2][1]), (corner[3][0], corner[3][1]), (0, 255, 255), 2)
    cv2.line(img, (corner[3][0], corner[3][1]), (corner[1][0], corner[1][1]), (0, 255, 255), 2)
    cv2.line(img, (corner[0][0], corner[0][1]), (corner[1][0], corner[1][1]), (0, 255, 255), 2)

    wid = tl.computeDist(corner[0][0], corner[0][1], corner[2][0], corner[2][1])
    hei = tl.computeDist(corner[0][0], corner[0][1], corner[1][0], corner[1][1])
    if wid < hei:
        tmp = hei
        hei = wid
        wid = tmp

    global cameraHeight
    shoewid = 0.0
    if cameraHeight > 0:
        shoewid = cameraHeight * wid / 3038.3
    print 'getContourShoe3(), shoe wid pixel = ', wid, ', shoe wid = ', shoewid * 10
    cv2.imwrite('foot.jpg', img)

    return
# find the white paper region
def detectEdge(img_orig, inx, isel):
    h, w, _ = img_orig.shape

    img = cv2.GaussianBlur(img_orig, (35, 35), 0)
    imgCanny, imgCannyDil = tl.getCanny(img, 5, 7, 1, 13, 13, 7, 7, False)
    #img = cv2.GaussianBlur(img_orig, (55, 55), 0)
    #imgCanny, imgCannyDil = tl.getCanny(img, 5, 7, 1, 13, 13, 7, 7, False)
    
    cv2.imwrite('startcardcanny.jpg', imgCanny)
    cv2.imwrite('startcardcannydil.jpg', imgCannyDil)
    cardContour = getContourCard(img_orig.copy(), imgCannyDil.copy())

    #return None
    if cardContour is None:
        print 'detectEdge(), Get card region none'
        return None
    #getContourShoe3(img_orig.copy())
    # card get longedge direction
    #return None

    getContourShoe(img_orig.copy(), imgCannyDil.copy())
    '''
    img = cv2.GaussianBlur(img_orig, (55, 55), 0)
    imgCanny, imgCannyDil = tl.getCanny(img, 15, 2, 1, 11, 11, 5, 5, False)
    getContourShoe2(img_orig.copy(), imgCannyDil.copy())
    '''

    return None


if __name__ == "__main__":
    im_names = []
    path = 'img/'
    files = os.listdir(path)
    for f in files:
        if (os.path.isfile(path + f)):
            im_names.append(path + f)
    cnt = 0
    listpic = []
    version = cv2.__version__
    for imgname in im_names:
        print 'pic name = ', imgname
        cnt += 1
        img = cv2.imread(imgname, 1)

        warpDst = detectEdge(img, cnt, 1)

        if warpDst is None:
            print 'fail to get white paper'
            continue
        if version == '3.1.0':
            meter = 1.0

        else:
            meter = 1.0
            print 'shoe Length = ', meter

