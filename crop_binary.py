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

nPageCnt = 0

def binproc(img):

    (b,g,r) = cv2.split(img)


    cv2.imwrite('r.jpg', r)
    _, imgThreshr = cv2.threshold(r, 200, 255, cv2.THRESH_BINARY)
    cv2.imwrite('threshr.jpg', imgThreshr)

    cv2.imwrite('g.jpg', g)
    _, imgThreshg = cv2.threshold(g, 200, 255, cv2.THRESH_BINARY)
    cv2.imwrite('threshg.jpg', imgThreshg)


    cv2.imwrite('b.jpg', b)
    _, imgThreshb = cv2.threshold(b, 200, 255, cv2.THRESH_BINARY)
    cv2.imwrite('threshb.jpg', imgThreshb)

    return imgThreshr


if __name__ == '__main__':

    global image

    nPageCnt = 0
    filenames = [img for img in glob.glob("clrproc/*.jpg")]

    # filenames.sort() # ADD THIS LINE
    # print 'size = ', filenames.count

    for imgName in filenames:
        print 'name = ', imgName[8:-4]

        imgOrign = cv2.imread(imgName, 1)
        h, w= imgOrign.shape[:2]
        
        pagename = './crop/'+imgName[8:-4]+'.jpg'
        dst = binproc(imgOrign)
        cv2.imwrite(pagename, dst)
        break
    
    print 'end...'