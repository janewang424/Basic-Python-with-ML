# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 02:20:01 2020

@author: janew
"""


import cv2
import numpy as np
from paddle import *
if __name__=='__main__':
    img = cv2.imread('./images.jpg')
    pan = cv2.imread('./panda.jpg')
    print(img.shape,pan.shape)
    pan=cv2.resize(pan,dsize=(333,151))
    print(img.shape, pan.shape)
    mix=cv2.addWeighted(img,0.7,pan,0.3,1)
    cv2.imshow('img',mix)
    cv2.imwrite('img-panda.jpg',mix)
    cv2.waitKey(0)
