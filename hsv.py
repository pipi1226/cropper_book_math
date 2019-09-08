# -*- coding: utf-8 -*-
import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

def hsv_mask(image, lower_color, upper_color, kernel_size):
    lower = np.array(lower_color, np.uint8)
    upper = np.array(upper_color, np.uint8)
    # 得到二值图像，在lower~upper范围内的值为255，不在的值为0
    mask = cv2.inRange(image, lower, upper)
    # 进行腐蚀和膨胀
    if kernel_size:
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        dilated = cv2.dilate(mask, kernel)
        eroded = cv2.erode(dilated, kernel) 
        return eroded
    else:
        return mask

if __name__ == '__main__':
    imagePath = './0.jpg'
    im= cv2.imread(imagePath,1)
    # 模糊，消除噪声
    blur_image = cv2.GaussianBlur(im, (5, 5), 0)
    # 转换HSV颜色空间 
    hsv_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)

    # 蓝色区域的HSV值范围
    hsv_low = [105, 80, 80]
    hsv_high = [125, 180, 140]

    # 通过设置的范围，去除不要的区域
    kernel_size = (20,20)
    img = hsv_mask(hsv_image, hsv_low, hsv_high, kernel_size)

    # 找连通域
    labels = measure.label(img, connectivity=2)
    pro = measure.regionprops(labels)

    # 画矩形框
    fig, ax = plt.subplots()
    im = im[:, :, (2, 1, 0)]
    ax.imshow(im, aspect='equal')
    for region in pro:
        box = region.bbox    
        ax.add_patch(
                plt.Rectangle((box[1], box[0]),
                            box[3] - box[1],
                            box[2] - box[0], fill=False,
                            edgecolor='red', linewidth=2)
            )
    # 去除坐标
    plt.axis('off')

    #dpi是设置清晰度的，大于300就很清晰了
    plt.savefig('result.png', dpi=300)
 