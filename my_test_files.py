# -*-coding: utf-8 -*-
"""
    @Project: AestheticsAnalysis
    @File   : my_test_files.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-07-20 09:48:15
"""
import h5py
import os
import cv2
import math
import numpy as np
import random

img = cv2.imread('test_image/02123_0.39_0.76.jpg')
img = cv2.resize(img, (200, 300))
cv2.imshow(img)
img = img.transpose(2, 0, 1)