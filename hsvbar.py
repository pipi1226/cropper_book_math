# -*- coding: utf-8 -*-
import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

# callbacks
def nothing(x):
    pass

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

def detectWhiteHSV(imgAdd, HMin, SMin, VMin, HMax, SMax, VMax):
    #img = cv2.imread(im_name, 1)

    # img = cv2.blur(img, (5,5))
    img = cv2.GaussianBlur(imgAdd, (5, 5), 0)

    img2hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img2hsv)

    # img2hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.equalizeHist()
    channels = np.zeros(img.shape, np.uint8)
    imgNhsv = img2hsv.copy()

    cv2.equalizeHist(v, v)
    imgNhsv = cv2.merge((h, s, v))

    lower_white = np.array([HMin, SMin, VMin])
    upper_white = np.array([HMax, SMax, VMax])

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(imgNhsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    # res = cv2.bitwise_and(h, h, mask= mask)
    res = cv2.bitwise_and(img, img, mask=mask)

    return res

if __name__ == '__main__':
    imagePath = './1.jpg'
    img= cv2.imread(imagePath,1)

    cv2.namedWindow('Track Bar', cv2.WINDOW_NORMAL)
    
    # creat track bars
    cv2.createTrackbar('H_Min', 'Track Bar', 0, 255, nothing)
    cv2.createTrackbar('S_Min', 'Track Bar', 0, 255, nothing)
    cv2.createTrackbar('V_Min', 'Track Bar', 0, 255, nothing)

    cv2.createTrackbar('H_Max', 'Track Bar', 0, 255, nothing)
    cv2.createTrackbar('S_Max', 'Track Bar', 0, 255, nothing)
    cv2.createTrackbar('V_Max', 'Track Bar', 0, 255, nothing)

    kernel_size = (20,20)

    # while loop
    while(1):
        HMin = cv2.getTrackbarPos('H_Min', 'Track Bar')
        SMin = cv2.getTrackbarPos('S_Min', 'Track Bar')
        VMin = cv2.getTrackbarPos('V_Min', 'Track Bar')

        HMax = cv2.getTrackbarPos('H_Max', 'Track Bar')
        SMax = cv2.getTrackbarPos('S_Max', 'Track Bar')
        VMax = cv2.getTrackbarPos('V_Max', 'Track Bar')

        # if HMin >= HMax:
        #     continue

        # if SMin >= SMax:
        #     continue

        # if VMin >= VMax:
        #     continue

        # 蓝色区域的HSV值范围
        hsv_low = [HMin, SMin, VMin]
        hsv_high = [HMax, SMax, VMax]

        # 通过设置的范围，去除不要的区域
        
        # imgHSV = hsv_mask(im, hsv_low, hsv_high, kernel_size)
        res = detectWhiteHSV(img, HMin, SMin, VMin, HMax, SMax, VMax)


        cv2.imshow('Track Bar', res)

        print "HMin = ", HMin, ", SMin = ", SMin, ", VMin = ", VMin
        print "HMax = ", HMax, ", SMax = ", SMax, ", VMax = ", VMax

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


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
 