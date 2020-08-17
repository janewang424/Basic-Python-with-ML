# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 03:21:24 2020

@author: janew
"""

import numpy as np
import cv2
if __name__ == '__main__':
    #Load image and mask
    img = cv2.imread('./friends.jpg')
    head = cv2.imread('./panda.jpg')
    face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
    img_gray = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
    head_gray = cv2.cvtColor(head, code=cv2.COLOR_BGR2GRAY)
    
    #Binarize the image, BlackWhite
    threshold,head_binary = cv2.threshold(head_gray, 50,255, cv2.THRESH_OTSU)
    
    #Binarize the image and search by the contour
    contours, hierarchy = cv2.findContours(head_binary,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    #Based on the contours, calculate the areas
    areas = []
    for contour in contours:
        areas.append(cv2.contourArea(contour))
    areas = np.asarray(areas)
    index = areas.argsort() #little to large, last but one, second largest contour area. The largest area is the whole pic.
    
    #mask, all black
    mask = np.zeros_like(head_gray,dtype=np.uint8)
    #Black pic draw white contour
    mask = cv2.drawContours(mask,contours,index[-2],(255,255,255),
                            thickness = -1)
    #Face detection
    faces = face_detector.detectMultiScale(img_gray)

    #Draw contours
    for x,y,w,h in faces:
        mask2 = cv2.resize(mask,(w,h))
        head2 = cv2.resize(head,(w,h))
        for i in range(h): #Exchange by pixel to pixel
            for j in range(w):
                if (mask2[i,j] == 255).all(): # If all 255, then it is contour
                    img[i + y,j + x] = head2[i,j]
        
        
        mask2 = cv2.resize(head,dsize = (w,h))
        img[y:y+h,x:x+w] = head2
    cv2.imshow('faces',img)
    #cv2.imwrite('face',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()