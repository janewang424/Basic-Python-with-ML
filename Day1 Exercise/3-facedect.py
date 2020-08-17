# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 02:35:29 2020

@author: janew
"""


import cv2
if __name__ == '__main__':
    img = cv2.imread('./friends.jpg')
    gray = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
    
    face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
    faces = face_detector.detectMultiScale(gray)
    print(faces)
    for x,y,w,h in faces:
        # cv2.rectangle(img,
        #               pt1 = (x,y), #left top
        #               pt2 = (x+w,y+h), #right top
        #               color = [0,0,255], #red,BGR
        #               thickness = 2) #line thickness
        cv2.circle(img,
                   center = (x+w//2,y+h//2),
                   radius = w//2,
                   color = [0,0,255],
                   thickness = 2)
    cv2.imshow('faces',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()