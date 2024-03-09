import numpy as np
import cv2
import json

f = open('annotations_train.json')

im = cv2.imread('000000000632.jpg',1)

im_f = cv2.bilateralFilter(im,9,75,75)
kernel = np.ones((7, 7), np.float32)/25

kernel2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
im_fi = cv2.filter2D(im_f, -1, kernel2)
img = im
im_fi = cv2.addWeighted(im_fi, 1.5, img, -0.5, 0, img);
cv2.imshow("im", img)
cv2.waitKey()