# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 03:07:30 2020

@author: janew
"""


import cv2
if __name__ == '__main__':
    img = cv2.imread('./friends.jpg')
    head = cv2.imread('./images.jpg')
    gray = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
    
    face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
    faces = face_detector.detectMultiScale(gray)
    print(faces)
    for x,y,w,h in faces:
        dog = cv2.resize(head,dsize = (w,h))
        img[y:y+h,x:x+w] = dog
    cv2.imshow('faces',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
