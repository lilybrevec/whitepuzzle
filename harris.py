#!/usr/bin/env python
# -*- coding: utf-8 -*
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

img_origin = cv2.imread('img/white_0_1.jpg')
hw = img_origin.shape[0] / img_origin.shape[1]
img = cv2.resize(img_origin, (200, int(200.0*hw)))

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
#dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.cornerHarris(gray, 4, 5, 0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst, None)

tmp = np.zeros(dst.shape[:2], np.uint8)
tmp[ dst > 0.01*dst.max() ] = [255]
tmp[ dst <= 0.01*dst.max() ] = [0]
plt.subplot(121),plt.imshow(tmp)
plt.title('CornerHarrisMask'), plt.xticks([]), plt.yticks([])

# Threshold for an optimal value, it may vary depending on the image.
img[ dst > 0.01*dst.max() ] = [0,0,255]
plt.subplot(122),plt.imshow(img)
plt.title('CornerHarris'), plt.xticks([]), plt.yticks([])

plt.show()
