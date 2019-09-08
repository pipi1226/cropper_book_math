# -*- coding: utf-8 -*-
import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import math

from inspect import currentframe, getframeinfo

frameinfo = getframeinfo(currentframe())

HMin = 0
HMax = 24

SMin = 0
SMax = 70

VMin  = 0
VMax = 255

nPageCnt = 0

def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno

def computeDist(x1, y1, x2, y2):
    meter = 0.0
    deltax = abs(x1-x2)
    deltay = abs(y1-y2)
    meter = math.sqrt(deltax*deltax + deltay*deltay)
    return meter

def findLargestContour(contours, h, w):
    size = w * h * 1.0
    print get_linenumber(), ',', type(size)
    maxArea = cv2.contourArea(contours[0])
    print get_linenumber(), ',', type(maxArea)
    inxMax = 0
    for cnt, cons in enumerate(contours):
        area = cv2.contourArea(cons)
        print get_linenumber(), ',', type(area)
        if (maxArea < area) and (area < size.any()):
            maxArea = area
            inxMax = cnt
            # x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    print 'max=', maxArea, 'inxMax = ', inxMax, 'maxpoints = ', len(contours[inxMax])
    return inxMax, contours[inxMax]

def warpPerspective(img, cntMax):
    
    hull = cv2.convexHull(cntMax)
    curve = np.int0(cntMax)

    #print 'warpPerspective len(hull)=',len(hull), ' hull = ', hull

    h, w, _ = img.shape
    # get 4 corner points
    midH = h/2
    midW = w/2
    #print 'midH=', midH, 'midW=', midW, 'h=', h, 'w=', w
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
        print 'arHull = ', arHull
        # down or up
        if arHull[1] < midH:
            # left or right
            if arHull[0] < midW:
                #print 'left up'
                tmpDist = computeDist(arHull[0], arHull[1], 0,0)
                #print 'left up tmpDist = ', tmpDist, 'arHull[0] = ', arHull[0], 'arHull[1] = ', arHull[1], 'X2=', 0, 'Y2=', 0, 'minLUDist=', minLUDist
                if minLUDist > tmpDist:
                    minLUDist = tmpDist
                    luCorner[0] = arHull[0]
                    luCorner[1] = arHull[1]

            else:
                #print 'right up'
                tmpDist = computeDist(arHull[0], arHull[1], w, 0)
                #print 'right up tmpDist = ', tmpDist, 'arHull[0] = ', arHull[0], 'arHull[1] = ', arHull[1],'X2=',w,'Y2=', 0, 'minLDDist=', minRUDist
                if minRUDist > tmpDist:
                    minRUDist = tmpDist
                    ruCorner[0] = arHull[0]
                    ruCorner[1] = arHull[1]
        else:
            # left or right
            if arHull[0] < midW:
                tmpDist = computeDist(arHull[0], arHull[1], 0, h)
                #print 'left down tmpDist = ', tmpDist, 'arHull[0] = ', arHull[0], 'arHull[1] = ', arHull[1], 'X2=',0,'Y2=', h, 'minLDDist=', minLDDist
                if minLDDist > tmpDist:
                    minLDDist = tmpDist
                    ldCorner[0] = arHull[0]
                    ldCorner[1] = arHull[1]
            else:
                #print 'right down'
                tmpDist = computeDist(arHull[0], arHull[1], w, h)
                #print 'right down tmpDist = ', tmpDist, 'arHull[0] = ', arHull[0], 'arHull[1] = ', arHull[1], 'X2=', w, 'Y2=', h, 'minRDDist=', minRDDist
                if minRDDist > tmpDist:
                    minRDDist = tmpDist
                    rdCorner[0] = arHull[0]
                    rdCorner[1] = arHull[1]


    #print 'lucorner = ', luCorner
    #print 'rucorner = ', ruCorner
    #print 'ldcorner = ', ldCorner
    #print 'rdcorner = ', rdCorner

    # compute four corner points
    bCompute = 0
    if rdCorner[0] > ruCorner[0] and abs(rdCorner[0] - ruCorner[0]) > 100:
        bCompute = 1
    if rdCorner[0] < ruCorner[0] and abs(rdCorner[0] - ruCorner[0]) > 100:
        bCompute = 2
    #print 'bCOmpute = ', bCompute
    if bCompute == 1:
        deltaY = ruCorner[1] - luCorner[1]
        deltaX = ruCorner[0] - luCorner[0]
        k = 1.0 * (deltaY / deltaX)
        b = ruCorner[1] - k * ruCorner[0]
        #print 'k=', k ,'b = ',b
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
        #print 'k=', k, 'b = ', b
        x = ruCorner[0]
        y = (int)(k * x + b)
        if y > h:
            y = h
        rdCorner[0] = x
        rdCorner[1] = y

    print 'lucorner = ', luCorner
    print 'rucorner = ', ruCorner
    print 'ldcorner = ', ldCorner
    print 'rdcorner = ', rdCorner

    imgCur = img.copy()
    cv2.drawContours(imgCur, [hull], 0, (0, 255, 255), -1)

    imgCurMask = np.full(imgCur.shape, 0, dtype=np.uint8)
    cv2.drawContours(imgCurMask, [hull], 0, (255, 255, 255), -1)
    imgNoCurMask = cv2.bitwise_not(imgCurMask)
    imgRes = img.copy()
    imgRes = cv2.bitwise_and(img, imgCurMask, imgNoCurMask)

    cv2.circle(imgRes, (luCorner[0], luCorner[1]), 5, (0,255,255), -1)
    cv2.circle(imgRes, (ldCorner[0], ldCorner[1]), 5, (0,255,255), -1)
    cv2.circle(imgRes, (ruCorner[0], ruCorner[1]), 5, (0,255,255), -1)
    cv2.circle(imgRes, (rdCorner[0], rdCorner[1]), 5, (0,255,255), -1)
    #cv2.imwrite('points.jpg', imgRes)

    # 4.perspective
    ptsPers1 = np.float32([[luCorner[0], luCorner[1]], [ldCorner[0], ldCorner[1]], [ruCorner[0], ruCorner[1]], [rdCorner[0], rdCorner[1]]])
    ptsPers2 = np.float32([[0, 0], [0, heiResize], [widResize, 0], [widResize, heiResize]])
    #print 'pstper1=', ptsPers1, 'pstper2=', ptsPers2
    M_perspective = cv2.getPerspectiveTransform(ptsPers1, ptsPers2)
    warpPersDst = cv2.warpPerspective(img, M_perspective, (widResize, heiResize))
    cv2.imwrite('perspec.jpg', warpPersDst)

    return warpPersDst

def judgeIsPageorNot(imgAdd):
    bRet = True
    h,w,c = imgAdd.shape

    imgBlur = cv2.GaussianBlur(imgAdd, (5, 5), 0)

    img2hsv = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img2hsv)

    # img2hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    imgGray = cv2.cvtColor(imgAdd, cv2.COLOR_BGR2GRAY)

    # cv2.equalizeHist()
    channels = np.zeros(imgAdd.shape, np.uint8)
    imgNhsv = img2hsv.copy()

    cv2.equalizeHist(v, v)
    imgNhsv = cv2.merge((h, s, v))

    lower_white = np.array([HMin, SMin, VMin])
    upper_white = np.array([HMax, SMax, VMax])

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(imgNhsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    # res = cv2.bitwise_and(h, h, mask= mask)
    imgAdd2 = imgAdd.copy()
    # cv2.imwrite("imgAdd1.jpg", imgAdd)
    res = cv2.bitwise_and(imgAdd2, imgAdd2, mask=mask)

    # get external contour
    img1 = res.copy()

    # 转化成灰度图
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
 
    # 利用Sobel边缘检测生成二值图
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=9)
    # 二值化
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # imgReGray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # imgThresh = cv2.adaptiveThreshold(imgReGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 膨胀、腐蚀
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (65,35))   # how to set the parameter of dilate and erode?
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))

    # 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)
 
    # 腐蚀一次，去掉细节
    erosion = cv2.erode(dilation, element1, iterations=1)
 
    # 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=7)


    region = []
    # contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    inxMax, contMax = findLargestContour(contours, h, w)

    imgBoxMask = np.full(imgAdd.shape, 0, dtype=np.uint8)
    imgNoBoxMask = cv2.bitwise_not(imgBoxMask)

    # imgOut = warpPerspective(res, contours[inxMax])
    imgMerge = cv2.merge([dilation2, dilation2, dilation2])
   
    imgOut = cv2.bitwise_and(imgAdd, imgMerge)
    # cv2.imwrite("imgAdd.jpg", imgAdd)
    # cv2.imwrite("contour.jpg", imgOut)
    return bRet, imgOut


if __name__ == '__main__':

    imagePath = './0.jpg'
    imgOrign= cv2.imread(imagePath,1)

    h, w= imgOrign.shape[:2]

    base_size=h+120,w+120,3
    base=np.zeros(base_size,dtype=np.uint8)
    cv2.rectangle(base,(0,0),(w+120,h+120),(0,0,0),30) # really thick white rectangle
    # base[60:h+60,60:w+60]=imgOrign
    # cv2.imwrite('border.jpg', base)
    # bRet, imgOut = judgeIsPageorNot(base)

    # pagename = './re'+str(nPageCnt)+'.jpg'
    # cv2.imwrite(pagename, imgOut)

    cap = cv2.VideoCapture('./mathbooks/1grade.mov')
    kernel_size = (20,20)
    nPageCnt = 0
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            basecpy = base.copy()
            basecpy[60:h+60,60:w+60]=frame
            # Display the resulting frame
            cv2.imshow('Frame',basecpy)
            bRet, imgOut = judgeIsPageorNot(basecpy)
            if(bRet):
                pagename = './bookresult/'+str(nPageCnt)+'.jpg'
                cv2.imwrite(pagename, imgOut)
                nPageCnt += 1
                

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        # Break the loop
        else: 
            break
        
    # When everything done, release the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()
 