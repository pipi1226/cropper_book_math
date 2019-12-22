# -*- coding: utf-8 -*-
import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import math

from inspect import currentframe, getframeinfo

import tools as tl

import glob

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

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
bSel = 0
mask = []


def click_and_crop(event, x, y, flags, param):
    
    # grab references to the global variables
	global refPt, cropping
    
    # imgCpy = image.copy()

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if(event == cv2.EVENT_LBUTTONDOWN):
		refPt = [(x, y)]
		cropping = True
 
	# check to see if the left mouse button was released
	elif(event == cv2.EVENT_LBUTTONUP):
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
        # draw a rectangle around the region of interest
		cv2.rectangle(imgCpy, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow('Frame', imgCpy)

    # elif(event == cv2.EVENT_MOUSEMOVE):
    #     cv2.rectangle(imgCpy, refPt[0], [(x,y)], (0, 255, 0), 2)
    #     cv2.imshow('Frame', imgCpy)
		
    # print 'display image rect.'

def click_and_crop_roi(event, x, y, flags, param):
    
    # grab references to the global variables
	global refPt, cropping, bSel, mask

    
    # imgCpy = image.copy()

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if(event == cv2.EVENT_RBUTTONUP):
            print 'right button'
	# check to see if the left mouse button was released
	elif(event == cv2.EVENT_LBUTTONUP):
	    # record the ending (x, y) coordinates and indicate that
	    # the cropping operation is finished
	    refPt.append((x, y))
		
        print 'display image rect.'

if __name__ == '__main__':

    global image

    filenames = [img for img in glob.glob("bookresult/*.jpg")]

    filenames.sort() # ADD THIS LINE

    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Frame", click_and_crop_roi)

    imagePath = './0.jpg'
    image= cv2.imread(imagePath,1)

    while True:
        # if(0 == bSel):
        #     cv2.imshow('Frame', imgOrign)
        # else:
        cv2.imshow('Frame', image)

        print 'display image origin.'

        key = cv2.waitKey(1) & 0xFF
            
        # if the 'r' key is pressed, reset the cropping region
        if(key == ord("r")):
            image = image.copy()
            bSel = 1       
        elif(key == ord('s')):
            bSel = 0
            pts = np.array([refPt], np.int32)
            pts = pts.reshape((-1, 1, 2))
            mask = np.zeros(image.shape, np.uint8)
            mask = cv2.polylines(mask, [pts], True, (255, 255, 255))
            mask2 = cv2.fillPoly(mask, [pts], (255, 255, 255))
            image1,contours, hierarchy = cv2.findContours(cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            ROIarea = cv2.contourArea(contours[0])
            ROI = cv2.bitwise_and(mask2, image)
            image = ROI

        # if the 'c' key is pressed, break from the loop
        elif(key == ord("c")):
            break


    # for imgName in filenames:
    #     print 'name = ', imgName
    #     img = cv2.read(imgName, 1)
    #     h, w= imgOrign.shape[:2]
        
    #     while True:
    #         cv2.imshow('Frame', img)
    #         key = cv2.waitKey(1) & 0xFF
            
    #         # if the 'r' key is pressed, reset the cropping region
    #         if(key == ord("r")):
    #             image = clone.copy()

    #         # if the 'c' key is pressed, break from the loop
    #         elif(key == ord("c")):
    #             break
    
    # Closes all the frames
    cv2.destroyAllWindows()
 
