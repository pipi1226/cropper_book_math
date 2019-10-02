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

def detectRGBHisto(imgAdd, HMin, SMin, VMin, HMax, SMax, VMax):
    #img = cv2.imread(im_name, 1)

    # img = cv2.blur(img, (5,5))
    img = cv2.GaussianBlur(imgAdd, (5, 5), 0)

    bgr_planes = cv2.split(img)
    histSize = 256
    histRange = (0, 256) # the upper boundary is exclusive
    accumulate = False
    b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
    g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
    r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)
    hist_w = 512
    hist_h = 400
    bin_w = int(round( hist_w/histSize ))
    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    for i in range(1, histSize):
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int(round(b_hist[i-1])) ),
                ( bin_w*(i), hist_h - int(round(b_hist[i])) ),
                ( 255, 0, 0), thickness=2)
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int(round(g_hist[i-1])) ),
                ( bin_w*(i), hist_h - int(round(g_hist[i])) ),
                ( 0, 255, 0), thickness=2)
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int(round(r_hist[i-1])) ),
                ( bin_w*(i), hist_h - int(round(r_hist[i])) ),
                ( 0, 0, 255), thickness=2)


    return histImage

def filterRGB(img, RMin, GMin, BMin, RMax, GMax, BMax):
    imgOrig = img.copy()
    # imgR, imgG, imgB = cv2.split(imgOrig)
    # threshR = cv2.threshold(imgR, RMin, RMax, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # threshG = cv2.threshold(imgG, GMin, GMax, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # threshB = cv2.threshold(imgB, BMin, BMax, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # imgMerge1 = cv2.merge((threshB, threshG, threshR))
    # imgMerge1 = cv2.merge((imgR, imgG, imgB))
    lower_white = np.array([RMin, GMin, BMin])
    upper_white = np.array([RMax, GMax, BMax])
    imgFilter = cv2.inRange(imgOrig, lower_white, upper_white)
    # imgOut = imgOrig - imgFilter
    return imgFilter

if __name__ == '__main__':
    imagePath = './114.jpg'
    img= cv2.imread(imagePath,1)

    cv2.namedWindow('Track Bar', cv2.WINDOW_NORMAL)
    
    # creat track bars
    cv2.createTrackbar('R_Min', 'Track Bar', 0, 255, nothing)
    cv2.createTrackbar('G_Min', 'Track Bar', 0, 255, nothing)
    cv2.createTrackbar('B_Min', 'Track Bar', 0, 255, nothing)

    cv2.createTrackbar('R_Max', 'Track Bar', 255, 255, nothing)
    cv2.createTrackbar('G_Max', 'Track Bar', 255, 255, nothing)
    cv2.createTrackbar('B_Max', 'Track Bar', 255, 255, nothing)

    kernel_size = (20,20)


    # while loop
    while(1):
        RMin = cv2.getTrackbarPos('R_Min', 'Track Bar')
        GMin = cv2.getTrackbarPos('G_Min', 'Track Bar')
        BMin = cv2.getTrackbarPos('B_Min', 'Track Bar')

        RMax = cv2.getTrackbarPos('R_Max', 'Track Bar')
        GMax = cv2.getTrackbarPos('G_Max', 'Track Bar')
        BMax = cv2.getTrackbarPos('B_Max', 'Track Bar')

        # if HMin >= HMax:
        #     continue

        # if SMin >= SMax:
        #     continue

        # if VMin >= VMax:
        #     continue

        # 蓝色区域的HSV值范围
        hsv_low = [RMin, GMin, BMin]
        hsv_high = [RMax, GMax, BMax]

        # 通过设置的范围，去除不要的区域
        
        # imgHSV = hsv_mask(im, hsv_low, hsv_high, kernel_size)
        # res = detectRGBHisto(img, RMin, GMin, BMin, RMax, GMax, BMax)
        res = filterRGB(img, RMin, GMin, BMin, RMax, GMax, BMax)

        cv2.imshow('Track Bar', res)

        print "RMin = ", RMin, ", GMin = ", GMin, ", BMin = ", BMin
        print "RMax = ", RMax, ", GMax = ", GMax, ", BMax = ", BMax

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
    # img = hsv_mask(hsv_image, hsv_low, hsv_high, kernel_size)
    

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
 