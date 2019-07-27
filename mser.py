from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('011.jpg')
mser = cv2.MSER_create(_min_area=300)
# mser = cv2.MSER_create(_min_area=500)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# regions, boxes = mser.detectRegions(gray)

imgR, imgG, imgB = cv2.split(img)
regions, boxes = mser.detectRegions(imgR)

for box in boxes:
    x, y, w, h = box
    cv2.rectangle(img, (x,y),(x+w, y+h), (255, 0, 0), 2)

plt.imshow(img,'brg')
plt.show()
cv2.imwrite('result.jpg', img)