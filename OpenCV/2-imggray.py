# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 02:20:01 2020

@author: janew
"""


import cv2
import numpy as np
from paddle import *
if __name__ == '__main__':
    fax = cv2.imread('./images.jpg')
    print(fax.shape)
    print(fax)
    gray = cv2.cvtColor(fax,code = cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    threshold,dst = cv2.threshold(gray,100,255,type = cv2.THRESH_BINARY)
    cv2.imshow('fax',gray)
    cv2.imshow('Binary',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
