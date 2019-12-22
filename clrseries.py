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

def clrproc(img):

    res = np.float32(img)

    print type(img)
    print type(res)

    draw = cv2.GaussianBlur(res, (101, 101), 0)
    # draw = cv2.GaussianBlur(img, (101, 101), 0)

    drawnp = np.float32(draw)
    dst = (res/drawnp)*255
    # dst = (img/draw)*255

    # cv2.imwrite('draw.jpg', dst)
    return dst


if __name__ == '__main__':

    global image

    nPageCnt = 0
    filenames = [img for img in glob.glob("cut/*.jpg")]

    filenames.sort() # ADD THIS LINE
    # print 'size = ', filenames.count

    for imgName in filenames:
        print 'name = ', imgName[4:-4]

        imgOrign = cv2.imread(imgName, 1)
        h, w= imgOrign.shape[:2]
        
        pagename = './clrproc/'+imgName[4:-4]+'.jpg'
        dst = clrproc(imgOrign)
        cv2.imwrite(pagename, dst)
    
    print 'end...'