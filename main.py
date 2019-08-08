# code:utf-8

import cv2

img = cv2.imread("img/DSC_0003.JPG",1)

cv2.imshow('canny demo', img)

print(cv2.__version__)
