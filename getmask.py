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

def click_and_crop_roi(event, x, y, flags, param):
    
    # grab references to the global variables
	global refPt, cropping, bSel, mask, bFirst, prev, first, last

	if(event == cv2.EVENT_LBUTTONDOWN):
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
        print 'display image rect.'

if __name__ == '__main__':

    global image

    filenames = [img for img in glob.glob("cut/*.jpg")]

    filenames.sort() # ADD THIS LINE

    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Frame", click_and_crop_roi)

    imagePath = './0.jpg'
    image= cv2.imread(imagePath,1)
    imgOrigin = image.copy()

    while True:
        # if(0 == bSel):
        #     cv2.imshow('Frame', imgOrign)
        # else:
        # cv2.imshow('Frame', image)
        
        print 'display image origin.'

        key = cv2.waitKey(1) & 0xFF
            
        # if the 'r' key is pressed, reset the cropping region
        if(key == ord('r')):
            image = image.copy()
            bSel = 1
            first = (0,0)
            bFirst = 0
        elif(key == ord('s')):
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
            cv2.imwrite('ss.jpg', image)
        # if the 'c' key is pressed, break from the loop
        elif(key == ord('q')):
            break

    cv2.destroyAllWindows()
 
