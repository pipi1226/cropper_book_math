#!/usr/bin/env python
# --------------------------------------------------------
# FootDetect
# 2016-08-23
# a4 210mm x 297mm
# 2016-08-29
# get the shoe length
# 2016-08-31
# Perspective transformation rectify the rectangle
# --------------------------------------------------------

import os, sys, cv2
from matplotlib import pyplot as plt
import numpy as np

import tools as tl

import time

listpic = []

version = ''

widResize = 297 * 2
heiResize = 210 * 2

EPSILONG = 0.0000001
INFINIT = 999999

def judgeContourIsCard(contourCard, areaRect):
    bIsCard = True
    areaReal = cv2.contourArea(contourCard)
    extent = areaReal * 1.0 / areaRect

    # extent - 0.75 > EPSILONG result is not good
    if (extent - 0.5) > EPSILONG:
        bIsCard = True
    else:
        bIsCard = False
    return bIsCard

# get shoe length
def getShoeLength(imgShoe, curve):
    meter = 0.0
    print '------------------ getShoeLength() -----------------'
    imgCurMask = np.full(imgShoe.shape, 0, dtype=np.uint8)
    cv2.drawContours(imgCurMask, [curve], 0, (255, 255, 255), -1)
    imgNoCurMask = cv2.bitwise_not(imgCurMask)
    imgRes = imgShoe.copy()
    imgRes = cv2.bitwise_and(imgShoe, imgCurMask, imgNoCurMask)
    listpic.append(imgRes)
    cv2.imwrite('whitepaper.jpg', imgRes)

    imgRes = cv2.cvtColor(imgRes, cv2.COLOR_BGR2GRAY)
    _, imgThresh = cv2.threshold(imgRes, 1, 255, cv2.THRESH_BINARY)

    # dilate
    # dilate mask = (3, 3)
    dilatekenel = np.ones((13, 13), np.uint8)
    imgThresh = cv2.dilate(imgThresh, dilatekenel, iterations=1)

    listpic.append(imgThresh)
    cv2.imwrite('binpaper.jpg', imgThresh)
    thH, thW = imgThresh.shape[0], imgThresh.shape[1]
    print 'TH = ', thH, thW
    # horizontal projection
    hPro = [0] * (thH)

    # vertical projection
    vPro = [0] * (thW)

    for inxH in range(thH - 1):
        for inxW in range(thW - 1):
            # print inxH, inxW
            if imgThresh[inxH, inxW] == 255:
                hPro[inxH] += 1
                vPro[inxW] += 1
    # print hPro, len(hPro)

    # print vPro, len(vPro)

    theta = 20
    tsW = 0
    maxH = 0
    for inxW in range(thW - 1):
        if maxH < vPro[inxW]:
            maxH = vPro[inxW]

    for inxW in range(thW - 1):
        if abs(vPro[inxW] - maxH) < theta:
            tsW += 1

    meter = 1.0 * (tsW * 210) / thH
    meter = 297 - meter
    print 'getShoeLength() = ', meter
    return meter


# warp Perspective transformation
def warpPerspective(img, cntMax):
    hull = cv2.convexHull(cntMax)
    curve = np.int0(cntMax)

    # print 'warpPerspective len(hull)=',len(hull), ' hull = ', hull

    h, w, _ = img.shape
    # get 4 corner points
    midH = h / 2
    midW = w / 2
    # print 'midH=', midH, 'midW=', midW, 'h=', h, 'w=', w
    maxLen = max(h, w)
    minLUDist = (int)(maxLen)
    minRUDist = minLUDist
    minLDDist = minLUDist
    minRDDist = minLUDist
    luCorner = [0, 0]
    ruCorner = [0, 0]
    ldCorner = [0, 0]
    rdCorner = [0, 0]
    for inx, enHull in enumerate(hull):
        arHull = enHull[0]
        #print 'arHull = ', arHull
        # down or up
        if arHull[1] < midH:
            # left or right
            if arHull[0] < midW:
                # print 'left up'
                tmpDist = tl.computeDist(arHull[0], arHull[1], 0, 0)
                # print 'left up tmpDist = ', tmpDist, 'arHull[0] = ', arHull[0], 'arHull[1] = ', arHull[1], 'X2=', 0, 'Y2=', 0, 'minLUDist=', minLUDist
                if minLUDist > tmpDist:
                    minLUDist = tmpDist
                    luCorner[0] = arHull[0]
                    luCorner[1] = arHull[1]

            else:
                # print 'right up'
                tmpDist = tl.computeDist(arHull[0], arHull[1], w, 0)
                # print 'right up tmpDist = ', tmpDist, 'arHull[0] = ', arHull[0], 'arHull[1] = ', arHull[1],'X2=',w,'Y2=', 0, 'minLDDist=', minRUDist
                if minRUDist > tmpDist:
                    minRUDist = tmpDist
                    ruCorner[0] = arHull[0]
                    ruCorner[1] = arHull[1]
        else:
            # left or right
            if arHull[0] < midW:
                tmpDist = tl.computeDist(arHull[0], arHull[1], 0, h)
                # print 'left down tmpDist = ', tmpDist, 'arHull[0] = ', arHull[0], 'arHull[1] = ', arHull[1], 'X2=',0,'Y2=', h, 'minLDDist=', minLDDist
                if minLDDist > tmpDist:
                    minLDDist = tmpDist
                    ldCorner[0] = arHull[0]
                    ldCorner[1] = arHull[1]
            else:
                # print 'right down'
                tmpDist = tl.computeDist(arHull[0], arHull[1], w, h)
                # print 'right down tmpDist = ', tmpDist, 'arHull[0] = ', arHull[0], 'arHull[1] = ', arHull[1], 'X2=', w, 'Y2=', h, 'minRDDist=', minRDDist
                if minRDDist > tmpDist:
                    minRDDist = tmpDist
                    rdCorner[0] = arHull[0]
                    rdCorner[1] = arHull[1]

    # print 'lucorner = ', luCorner
    # print 'rucorner = ', ruCorner
    # print 'ldcorner = ', ldCorner
    # print 'rdcorner = ', rdCorner

    # compute four corner points
    bCompute = 0
    if rdCorner[0] > ruCorner[0] and abs(rdCorner[0] - ruCorner[0]) > 100:
        bCompute = 1
    if rdCorner[0] < ruCorner[0] and abs(rdCorner[0] - ruCorner[0]) > 100:
        bCompute = 2
    # print 'bCOmpute = ', bCompute
    if bCompute == 1:
        deltaY = ruCorner[1] - luCorner[1]
        deltaX = ruCorner[0] - luCorner[0]
        k = 1.0 * (deltaY / deltaX)
        b = ruCorner[1] - k * ruCorner[0]
        # print 'k=', k ,'b = ',b
        x = rdCorner[0]
        y = (int)(k * x + b)
        if y < 0:
            y = 0
        ruCorner[0] = x
        ruCorner[1] = y

    elif bCompute == 2:
        deltaY = rdCorner[1] - ldCorner[1]
        deltaX = rdCorner[0] - ldCorner[0]
        k = 1.0 * (deltaY / deltaX)
        b = rdCorner[1] - k * rdCorner[0]
        # print 'k=', k, 'b = ', b
        x = ruCorner[0]
        y = (int)(k * x + b)
        if y > h:
            y = h
        rdCorner[0] = x
        rdCorner[1] = y
    '''
    print 'lucorner = ', luCorner
    print 'rucorner = ', ruCorner
    print 'ldcorner = ', ldCorner
    print 'rdcorner = ', rdCorner
    '''

    imgCur = img.copy()
    cv2.drawContours(imgCur, [hull], 0, (0, 255, 255), -1)
    listpic.append(imgCur)

    imgCurMask = np.full(imgCur.shape, 0, dtype=np.uint8)
    cv2.drawContours(imgCurMask, [hull], 0, (255, 255, 255), -1)
    imgNoCurMask = cv2.bitwise_not(imgCurMask)
    imgRes = img.copy()
    imgRes = cv2.bitwise_and(img, imgCurMask, imgNoCurMask)

    imgRes = img.copy()
    imgRes = cv2.bitwise_and(img, imgCurMask, imgNoCurMask)
    listpic.append(imgRes)
    cv2.circle(imgRes, (luCorner[0], luCorner[1]), 5, (0, 255, 255), -1)
    cv2.circle(imgRes, (ldCorner[0], ldCorner[1]), 5, (0, 255, 255), -1)
    cv2.circle(imgRes, (ruCorner[0], ruCorner[1]), 5, (0, 255, 255), -1)
    cv2.circle(imgRes, (rdCorner[0], rdCorner[1]), 5, (0, 255, 255), -1)
    cv2.imwrite('points.jpg', imgRes)

    # 4.perspective
    ptsPers1 = np.float32([[luCorner[0], luCorner[1]], [ldCorner[0], ldCorner[1]], [ruCorner[0], ruCorner[1]],
                           [rdCorner[0], rdCorner[1]]])
    ptsPers2 = np.float32([[0, 0], [0, heiResize], [widResize, 0], [widResize, heiResize]])
    # print 'pstper1=', ptsPers1, 'pstper2=', ptsPers2
    M_perspective = cv2.getPerspectiveTransform(ptsPers1, ptsPers2)
    warpPersDst = cv2.warpPerspective(img, M_perspective, (widResize, heiResize))

    return warpPersDst


# get foot print

def getFootPrint_310(img):
    meter = 0.0

    image = cv2.pyrMeanShiftFiltering(img, 25, 10)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    imgThresh = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    h, w, c = img.shape
    print w, h, c

    edgeTh = 100
    imgCanny = cv2.Canny(imgThresh, edgeTh, edgeTh * 3, 1)
    '''
    # dilate mask = (3, 3)
    dilatekenel = np.ones((13, 13), np.uint8)
    dilation = cv2.dilate(imgCanny, dilatekenel, iterations=1)

    # erode mask = (3, 3)
    # dilatekernel = np.ones((13, 13), np.uint8)
    erokernel = np.ones((13, 13), np.uint8)
    erosion = cv2.erode(dilation, erokernel, iterations=1)

    imgCanny = (255 - erosion)
    '''

    imgCanny = (255 - imgCanny)

    _, contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    inxMax, contMax = tl.findLargestContour(contours, h, w)
    listpic.append(img)

    imgShoe = warpPerspective(img, contours[inxMax])

    h, w, _ = imgShoe.shape

    listpic.append(imgShoe)

    # preprocessing
    imgRes = cv2.cvtColor(imgShoe, cv2.COLOR_BGR2GRAY)

    edgeTh = 30
    imgCanny = cv2.Canny(imgRes, edgeTh, edgeTh * 3, 1)
    listpic.append(imgCanny)

    # dilate mask = (3, 3)
    dilatekenel = np.ones((17, 17), np.uint8)
    dilation = cv2.dilate(imgCanny, dilatekenel, iterations=1)

    # erode mask = (3, 3)
    # dilatekernel = np.ones((13, 13), np.uint8)
    erokernel = np.ones((13, 13), np.uint8)
    erosion = cv2.erode(dilation, erokernel, iterations=1)

    imgCanny = (255 - erosion)

    _, contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    inxMax, contMax = tl.findLargestContour(contours, h, w)

    meter = getShoeLength(imgShoe, contMax)

    return meter

# get the largest contour on the image
def getContour(img, thresh):
    boxes = []
    imgcanny = thresh.copy()

    if version == '3.1.0':
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    print 'Contours = ', len(contours)
    w, h, s = img.shape
    size = w * h

    chooseCnt = -1
    minRatio = INFINIT

    corner = []
    longEdge = 0
    shortEdge = 0
    bCardRot = 0
    bRot = 0
    smallestArea = w * h * 0.001
    maxArea = -1

    for cnt, cons in enumerate(contours):
        area, _, wid, hei = tl.computeRectArea(cons, version)
        if area <= smallestArea:
            continue
        bRot = 0
        if wid < hei:
            maxLen = hei
            hei = wid
            wid = maxLen
            bRot = 1

        ratio = 1.0 * (wid / hei)

        if abs(ratio - 1.41) < abs(minRatio - 1.41) and maxArea < area:
            # if abs(ratio - 1.57) < abs(minRatio - 1.57):
            #print 'getContourCard(), cnt=', cnt, 'ratio = ', ratio, 'area = ', area, 'minArea = ', fMinArea, 'fMaxArea = ', fMaxArea
            minRatio = ratio
            chooseCnt = cnt
            bCardRot = bRot
            maxArea = area

    if chooseCnt == -1:
        print 'getContourCard(), couldnot find card contour!!!.....'

    #print 'getContourCard(), cnt = ', chooseCnt, 'ratio = ', minRatio
    area, corner, wid, hei = tl.computeRectArea(contours[chooseCnt], version)
    # save card region jpg

    imgTest = img.copy()
    cv2.drawContours(imgTest, contours, chooseCnt, (0, 0, 255), 2)

    strPath = 'wholepaper.jpg'
    cv2.imwrite(strPath, imgTest)
    '''
    imgCurMask = np.full(img.shape, 0, dtype=np.uint8)
    cv2.drawContours(imgCurMask, contours, chooseCnt, (255, 255, 255), -1)
    imgNoCurMask = cv2.bitwise_not(imgCurMask)
    imgRes = img.copy()
    imgRes = cv2.bitwise_and(img, imgCurMask, imgNoCurMask)
    cv2.imwrite('whitepaper.jpg', imgRes)
    '''

    #print 'getContourCard(), corner = ', corner

    if len(corner) == 0:
        print 'getContourCard(), corner is none'
        return None

    corner = tl.sortBoxPoints(corner)
    print 'getContourCard(), after sort corner = ', corner
    longEdge = tl.computeDist(corner[0][0], corner[0][1], corner[1][0], corner[1][1])
    shortEdge = tl.computeDist(corner[0][0], corner[0][1], corner[2][0], corner[2][1])

    if longEdge < shortEdge:
        longEdge, shortEdge = shortEdge, longEdge
        pts1 = np.float32([[corner[0][0], corner[0][1]], [corner[1][0], corner[1][1]], [corner[2][0], corner[2][1]]])
        #pts2 = np.float32([[0, 0], [0, (int)(shortEdge)], [(int)(longEdge), 0]])
        pts2 = np.float32([[0, 0], [0, (int)(heiResize)], [(int)(widResize), 0]])
    else:
        pts1 = np.float32([[corner[0][0], corner[0][1]], [corner[2][0], corner[2][1]], [corner[1][0], corner[1][1]]])
        #pts2 = np.float32([[0, 0], [0, (int)(shortEdge)], [(int)(longEdge), 0]])
        pts2 = np.float32([[0, 0], [0, (int)(heiResize)], [(int)(widResize), 0]])

    #warpDst = tl.getWarpAffine(imgRes, pts1, pts2, (int)(longEdge), (int)(shortEdge))
    warpDst = tl.getWarpAffine(img, pts1, pts2, (int)(widResize), (int)(heiResize))
    cv2.imwrite('warpAffine.jpg', warpDst)
    return warpDst

    thH, thW = warpDst.shape[0], warpDst.shape[1]
    meter = 0.0
    _, imgThresh = cv2.threshold(warpDst, 1, 255, cv2.THRESH_BINARY)
    # dilate
    # dilate mask = (3, 3)

    cv2.imwrite('binpaper.jpg', imgThresh)

    print 'TH = ', thH, thW

    # horizontal projection
    hPro = [0] * (thH)

    # vertical projection
    vPro = [0] * (thW)
    minH = 999
    for inxH in range(thH - 1):
        bFirst = 1
        iFirst = 0
        for inxW in range(thW - 1):
            #print 'thresh = ', inxH, inxW, imgThresh[inxH, inxW]
            if imgThresh[inxH, inxW] == 255:
                if bFirst == 1:
                    iFirst = inxW
                    bFirst = 0
                else:
                    hPro[inxH] = inxW - iFirst
        if hPro[inxH] < minH:
            minH = hPro[inxH]
    tsW = minH
    print 'tsw = ', tsW
    meter = 1.0 * (tsW * 210) / thH
    meter = 297 - meter
    print 'getShoeLength() = ', meter
    #return meter
    return warpDst

# find the white paper region
def detectEdge(img_orig, img, inx, isel):
    h, w, _ = img_orig.shape

    img = cv2.GaussianBlur(img_orig, (35, 35), 0)
    #imgrui = cv2.Laplacian(img, cv2.CV_32F, ksize = 3)
    #cv2.imwrite('laplacian.jpg', imgrui)
    imgCanny, imgCannyDil = tl.getCanny(img, 10, 5, 1, 13, 13, 7, 7, False)
    warpDst = getContour(img_orig, imgCannyDil)

    return warpDst


if __name__ == "__main__":
    im_names = []
    # /home/sky/PycharmProjects/FootDetect/img
    # path = '/home/sky/PycharmProjects/FootDetect/img/'
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
        time1 = time.time()
        warpDst = detectEdge(img, None, cnt, 1)
        if warpDst is None:
            print 'could not get warpaffine picture'
            time2 = time.time()
            print 'process time = ', time2 - time1
            continue
        '''
        time2 = time.time()
        print 'process time = ', time2 - time1
        continue
        '''
        if version == '3.1.0':
            meter = 1.0
            meter = getFootPrint_310(warpDst)
        else:
            meter = 1.0
            meter = getFootPrint_2413(warpDst)
            print 'shoe Length = ', meter
        time2 = time.time()
        print 'process time = ', time2 - time1



