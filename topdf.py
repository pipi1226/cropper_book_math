# -*- coding: utf-8 -*-
import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import math

from inspect import currentframe, getframeinfo

import tools as tl

import glob

from PIL import Image

from fpdf import FPDF
import img2pdf

frameinfo = getframeinfo(currentframe())

HMin = 0
HMax = 173

SMin = 0
SMax = 98

VMin  = 0
VMax = 209

nPageCnt = 0

def img2pdfGen(filenames):
    file = open('xiao3.pdf', "wb")
    cnt = 0
    filelist = []
    for imgName in filenames:
        imgOrign = Image.open(imgName)
        filelist.append(imgOrign.filename)
        # file.write(pdf_bytes) 
        # file.add_page(pdf_bytes)
        # imgOrign.close()
        cnt+=1
        print 'cnt=', cnt
        # if(cnt == 3):
        #     break
    pdf_bytes = img2pdf.convert(filelist) 
    file.write(pdf_bytes) 
    file.close()


def pilGen(filenames):
    images = []
    nPageCnt = 0
    for imgName in filenames:
        # print 'name = ', imgName[4:-4]
        print 'name = ', imgName
        imgOrign = Image.open(imgName)
        w, h = imgOrign.size
        print 'w = ', w, 'h=',h
        if(nPageCnt == 0):
            im1 = imgOrign.convert('RGB')      
            images.append(im1)  
            print 'append first'
            nPageCnt = nPageCnt + 1
            # im1.save('chu2.pdf')
        else:
            im2 = imgOrign.convert('RGB')
            images.append(im2)
            print 'image cnt = ', len(images)
            break
            # im2.save('chu2.pdf')
        # pagename = './'+imgName[18:-4]+'.pdf'
        # print 'outname=', pagename
        # im1.save(pagename)
        # dst = clrproc(imgOrign)
        # cv2.imwrite(pagename, imgOrign, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
        # break
    print 'image cnt = ', len(images)
    images[0].save('chu2.pdf', 'pdf', save_all=True, append_images=images[1:])

def fpdfGen(filenames):
    imagelist = []
    nPageCnt = 0
    pdf = FPDF()
    hmin = 4544
    for imgName in filenames:
        # print 'name = ', imgName[4:-4]
        print 'name = ', imgName
        imgOrign = Image.open(imgName)
        w, h = imgOrign.size
        print 'w = ', w, 'h=',h
        if(h>hmin):
            hmin = h
        pdf.add_page()
        pdf.image(imgName, 0, 0, float(w*0.264583), float(h*0.264583))
        break
    print 'hmin=',hmin
    print 'image cnt = ', len(imagelist)
    # im1.save('chu2.pdf', save_all = 1, append_images=imagelist)
    pdf.output("chu2.pdf")

if __name__ == '__main__':

    global image

    
    filenames = [img for img in glob.glob("./math_g8_cmpress/*.jpg")]

    filenames.sort() # ADD THIS LINE
    print 'size = ', filenames.count
    
    # pilGen(filenames)
    # fpdfGen(filenames)
    img2pdfGen(filenames)
    
    print 'end...'