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

# [20200118] cut page manually

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
bSel = 0
mask = []
prev = (0,0)
first = (0,0)
last = (0,0)
bFirst = 0
newpage=0

def click_and_crop_roi(event, x, y, flags, param):
    
    # grab references to the global variables
	global refPt, cropping, bSel, mask, bFirst, prev, first, last
        global newpage

	if(event == cv2.EVENT_LBUTTONDOWN):
            if(0 == newpage):
                return
            refPt.append((x, y))
            if(0 == bFirst):
                first = (x,y)
                bFirst = 1
                prev = first
            else:
                # prev = [(x,y)]
                last = (x,y)
            
            if(0 == bFirst):
                prev = (x,y)
            else:
                cv2.line(image, prev, (x,y), (0, 0, 255), 2)
                
                prev = (x,y)
            print 'left down'
	# check to see if the left mouse button was released
	elif(event == cv2.EVENT_LBUTTONUP):
	    # record the ending (x, y) coordinates and indicate that
	    # the cropping operation is finished
	    print 'left up'
        cv2.imshow('Frame', image)
        # print 'display image rect.'

def saveCutJPG(imgName):

    global image
    global refPt, cropping, bSel, mask, bFirst, prev, first, last
    global newpage

    image = cv2.imread(imgName, 1)
    h, w= image.shape[:2]
    pagename = './crop/'+imgName[8:-4]+'.jpg'
    
    imgOrigin = image.copy()

    first = (0,0)
    bFirst = 0
    newpage = 1
    refPt = []

    # cv2.imwrite(pagename, image)
    # return
    # return imgOrigin

    while True:
        # if(0 == bSel):
        #     cv2.imshow('Frame', imgOrign)
        # else:
        # cv2.imshow('Frame', image)
        

        key = cv2.waitKey(1) & 0xFF
            
        # if the 'r' key is pressed, reset the cropping region
        if(key == ord('d')):
            print 'display image origin.'
            cv2.imshow('Frame', image)
        elif(key == ord('r')):
            print 'reset image origin.'
            image = image.copy()
            bSel = 1
            first = (0,0)
            bFirst = 0
            newpage = 0
        elif(key == ord('s')):
            print 'save image origin.'
            image = cv2.line(image, last, first, (0, 0, 255), 2)
            bSel = 0
            pts = np.array([refPt], np.int32)
            pts = pts.reshape((-1, 1, 2))
            mask = np.zeros(imgOrigin.shape, np.uint8)
            mask = cv2.polylines(mask, [pts], True, (255, 255, 255))
            mask2 = cv2.fillPoly(mask, [pts], (255, 255, 255))
            contours, hierarchy = cv2.findContours(cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            ROIarea = cv2.contourArea(contours[0])
            ROI = cv2.bitwise_and(mask2, imgOrigin)
            image = ROI
            cv2.imwrite(pagename, image)
        # if the 'c' key is pressed, break from the loop
        elif(key == ord('q')):
            newpage = 0
            print "next picture..."
            break
        

if __name__ == '__main__':

    global image
    global newpage

    filenames = [img for img in glob.glob("clrproc/*.jpg")]

    # filenames.sort() # ADD THIS LINE

    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Frame", click_and_crop_roi)

    # imagePath = './0.jpg'
    # image= cv2.imread(imagePath,1)
    # imgOrigin = image.copy()

    for imgName in filenames:
        print imgName
        # break
        saveCutJPG(imgName)
        # cv2.imwrite('test.jpg', img)

    cv2.destroyAllWindows()
 
