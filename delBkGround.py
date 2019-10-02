# -*- coding: utf-8 -*-
import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import math

from inspect import currentframe, getframeinfo

import tools as tl

frameinfo = getframeinfo(currentframe())

HMin = 0
HMax = 173

SMin = 0
SMax = 98

VMin  = 0
VMax = 209

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
        if (maxArea < area) and (area < size):
            maxArea = area
            inxMax = cnt
            # x,y,w,h = cv2.boundingRect(cnt)
            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    print 'max=', maxArea, 'inxMax = ', inxMax, 'maxpoints = ', len(contours[inxMax])
    return inxMax, contours[inxMax]

def judgeIsPageorNot(imgAdd):
    bRet = True
    h,w,c = imgAdd.shape

    imgBlur = cv2.GaussianBlur(imgAdd, (5, 5), 0)

    lower_white = np.array([HMin, SMin, VMin])
    upper_white = np.array([HMax, SMax, VMax])

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(imgBlur, lower_white, upper_white)
    # Bitwise-AND mask and original image
    # res = cv2.bitwise_and(h, h, mask= mask)
    imgAdd2 = imgAdd.copy()
    # cv2.imwrite("imgAdd1.jpg", imgAdd)
    res = cv2.bitwise_and(imgAdd2, imgAdd2, mask=mask)

    # get external contour
    img1 = imgAdd.copy()

   
    # imgOut = cv2.bitwise_and(imgAdd, imgMerge)
    # cv2.imwrite("imgAdd.jpg", imgAdd)
    # cv2.imwrite("contour.jpg", imgOut)

    # imgOut = cv2.drawContours(imgAdd, contMax, -1, (255,255,255), 2)
    imgFore = img1 - res

    # filter skin color
    lower_skin = np.array([31, 42, 72])
    upper_skin = np.array([99, 108, 196])

    maskSkin = cv2.inRange(imgFore, lower_skin, upper_skin)
    resSkin = cv2.bitwise_and(imgFore, imgFore, mask=maskSkin)

    imgForePage = imgFore - resSkin

    gray = cv2.cvtColor(imgForePage, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (65,35))   # how to set the parameter of dilate and erode?
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))

    # 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)
 
    # 腐蚀一次，去掉细节
    erosion = cv2.erode(dilation, element1, iterations=1)
 
    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inxMax, maxCont = findLargestContour(contours, h, w)

    area = cv2.contourArea(maxCont)
    if(area < 999999.0):
        return 0, imgAdd

    x,y,w,h = cv2.boundingRect(maxCont)
    # cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),2)
    # imgOut = cv2.drawContours(img1, contours, inxMax, (0,0,255), 2)
    imgOut = img1[y-5:y+5+h, x-5:x+5+w]
    return bRet, imgOut

# 计算单通道的直方图的相似值 
def calculate(image1, image2): 
    hist1 = cv2.calcHist([image1],[0],None,[256],[0.0,255.0]) 
    hist2 = cv2.calcHist([image2],[0],None,[256],[0.0,255.0]) 
    # 计算直方图的重合度 
    degree = 0
    for i in range(len(hist1)): 
        if (hist1[i] != hist2[i]): 
            degree = degree + (1 - abs(hist1[i]-hist2[i])/max(hist1[i],hist2[i])) 
        else: 
            degree = degree + 1
            degree = degree/len(hist1) 
    return degree 

# 通过得到每个通道的直方图来计算相似度 
def classify_hist_with_split(image1, image2, size = (256,256)): 
    # 将图像resize后，分离为三个通道，再计算每个通道的相似值 
    image1 = cv2.resize(image1,size) 
    image2 = cv2.resize(image2,size) 
    sub_image1 = cv2.split(image1) 
    sub_image2 = cv2.split(image2) 
    sub_data = 0
    for im1,im2 in zip(sub_image1,sub_image2): 
        sub_data += calculate(im1,im2) 
        sub_data = sub_data/3
    return sub_data


def computeSimilar(imgPrev, imgNext):
    degree = classify_hist_with_split(imgPrev, imgNext, size = (256,256))
    print 'degree = ', degree
    if(degree >= 0.15):
        return 1
    else:
        return 0


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
    bCmpFirstRet = 0

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
                if(0 == bCmpFirstRet):
                    imgPrev = imgOut
                    imgCur = imgOut
                    bCmpFirstRet = 1
                    cv2.imwrite(pagename, imgOut)
                    cv2.imshow('Frame', imgOut)
                else:
                    bCmpRet = computeSimilar(imgPrev, imgOut)
                    if(0 == bCmpRet):
                        imgPrev = imgOut
                        cv2.imwrite(pagename, imgOut)
                        cv2.imshow('Frame', imgOut)
                
                # cv2.imwrite(pagename, imgOut)
                
                nPageCnt += 1
                # if cv2.waitKey(25) & 0xFF == ord('s'):
                #     cv2.imwrite(pagename, imgOut)
                

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
 