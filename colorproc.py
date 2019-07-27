# code: utf-8

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('011.JPG', 1)

res = np.float32(img) 

# minVal = np.amin(res)
# maxVal = np.amax(res)
# draw = cv2.convertScaleAbs(img, alpha=255.0/(maxVal - minVal), beta=-minVal * 255.0/(maxVal - minVal))

draw = cv2.GaussianBlur(res, (101, 101), 0)
dst = (res/draw)*255
cv2.imwrite('draw.jpg', dst)

re = (draw - dst)
cv2.imwrite('re.jpg', re)

print cv2.__version__

