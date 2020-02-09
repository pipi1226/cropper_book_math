# -*- coding: utf-8 -*-
# cite from wechat 

import cv2
import numpy as np

# def getHSV(img):


if __name__ == '__main__':
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    # cap = cv2.VideoCapture('./mathbooks/1grade.mov')
    cap = cv2.VideoCapture(0)
    nPageCnt = 0
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    cv2.namedWindow('canny demo', cv2.WINDOW_NORMAL)

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret,frame = cap.read()
        if ret == True:
        
            # Display the resulting frame
            cv2.imshow('Frame',frame)
        
            if(cv2.waitKey(25) & 0xFF == ord('w')):
                pagename = str(nPageCnt)+'.jpg'
                cv2.imwrite(pagename, frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        # Break the loop
        else: 
            print 'ret is false'
            break
    
    # When everything done, release the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()