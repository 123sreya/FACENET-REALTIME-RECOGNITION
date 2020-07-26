# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 22:16:46 2020

@author: sreya
"""

import cv2
from mtcnn.mtcnn import MTCNN
detector = MTCNN()
image = cv2.imread("billgatess.jfif")
result = detector.detect_faces(image)
print(result)