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

def tesscomp(img):
    
    max_line_width = dpi/kThinLineFraction
    min_line_len = dpi/kMinLineLengthFraction
    closing_brick = max_line_width / 3
    if(closing_brick %2):
        closing_brick += 1

    kernel = np.ones((closing_brick,closing_brick), np.uint8)
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    thopen1 = cv2.erode(th2, kernel, iterations=2)
    thopen = cv2.dilate(thopen1, kernel, iterations=1)

    return thopen


if __name__ == '__main__':

    global image

    nPageCnt = 0
    filenames = [img for img in glob.glob("cut/*.jpg")]

    # filenames.sort() # ADD THIS LINE
    # print 'size = ', filenames.count

    for imgName in filenames:
        print 'name = ', imgName[8:-4]
        imgName = "2021.jpg"
        imgOrign = cv2.imread(imgName, 0)
        h, w= imgOrign.shape[:2]
        
        # pagename = './crop20200118/'+imgName[8:-4]+'.jpg'
        pagename = './crop20200118/02.jpg'
        dst = tesscomp(imgOrign)
        cv2.imwrite(pagename, dst)

        break
    
    print 'end...'

