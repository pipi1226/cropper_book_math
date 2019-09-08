# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('0.JPG', 1)

# res = np.asarray(img, np.float32)
# cv2.imwrite('res.jpg', res)


res = np.float32(img)

# print type(img)
# print type(res)


# minVal = np.amin(res)
# maxVal = np.amax(res)
# draw = cv2.convertScaleAbs(img, alpha=255.0/(maxVal - minVal), beta=-minVal * 255.0/(maxVal - minVal))

draw = cv2.GaussianBlur(res, (101, 101), 0)
cv2.imwrite('draw.jpg', draw)

drawnp = np.float32(draw)
dst = (res/drawnp)*255

cv2.imwrite('dst.jpg', dst)

re = (drawnp - dst)
cv2.imwrite('re.jpg', re)


# binary threshold
# maxvalue = 211
# value = 73
# gaus = cv2.adaptiveThreshold(re, maxvalue, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, value, 1)

reInv = (255 - re)
cv2.imwrite('inverse.jpg', reInv)

reInv = cv2.imread('inverse.jpg', 1)
(b,g,r) = cv2.split(reInv)

cv2.imwrite('inverseR.jpg', r)
cv2.imwrite('inverseB.jpg', b)
cv2.imwrite('inverseG.jpg', g)


bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

hist = cv2.merge((bH, gH, rH))
cv2.imwrite("hist.jpg", rH)



_, imgThresh = cv2.threshold(hist, 10, 128, cv2.THRESH_BINARY)
cv2.imwrite('thresh.jpg', imgThresh)

imgCpy = reInv.copy()

_, contours, hierarchy = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cv2.drawContours(imgCpy,contours, -1,(255,0,255),1)

cv2.imwrite('adapt.jpg', imgCpy)


print cv2.__version__

