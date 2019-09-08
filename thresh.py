# code: utf-8

import cv2
import numpy as np
from matplotlib import pyplot as plt

import tools as tl

img = cv2.imread('0.jpg',1)

imgR, imgG, imgB = cv2.split(img)

gauss = cv2.GaussianBlur(imgR, (9, 9), 1)
# _, imgThresh = cv2.threshold(imgR, 90, 255, cv2.THRESH_BINARY)
_, imgThresh = cv2.threshold(imgR, 60, 255, cv2.THRESH_BINARY)
print 'cv2.BINARY = ', cv2.THRESH_BINARY
imgThreshBlur = cv2.GaussianBlur(imgThresh, (9, 9), 1)
# imgBin = cv2.
maxvalue = 211
value = 73
gaus = cv2.adaptiveThreshold(gauss, maxvalue, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, value, 1)
imgCpy = gauss.copy()
_, contours, hierarchy = cv2.findContours(gauss, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cv2.drawContours(imgCpy,contours, -1,(255,0,255),1)
cv2.imwrite('thresh.jpg', imgThresh)
cv2.imwrite('adapt.jpg', imgCpy)
cv2.imwrite('gaus.jpg', imgThreshBlur)
# imgCanny, imgCannyDil = tl.getCanny(img, 5, 7, 1, 9, 9, 5, 5, False)
# cv2.imwrite('shoelength.jpg', imgCannyDil)

# cv2.imshow('enhanced', imgCannyDil)
# cv2.waitKey(0)

print cv2.__version__

