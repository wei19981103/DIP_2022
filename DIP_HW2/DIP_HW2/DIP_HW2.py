import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(".\\lena.jpg", 1)
height, width, channel_num = img.shape
cv2.imshow('Original',img)

#gray 1
gray_img1 = np.zeros([height, width], np.uint8)

for i in range(height):
    for j in range(width):
        color = img[i][j]
        gray_img1[i][j] = color[0] / 3 + color[1] / 3 + color[2] / 3

cv2.imshow('Gray1', gray_img1)

#gray 2
gray_img2 = np.zeros([height, width], np.uint8)

for i in range(height):
    for j in range(width):
        color = img[i][j]
        gray_img2[i][j] =  round(0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0])

cv2.imshow('Gray2', gray_img2)

#gray difference
gray_dif_img = np.zeros([height, width], np.uint8)
for i in range(height):
    for j in range(width):
        gray_dif_img[i][j] = int(gray_img1[i][j]) - int(gray_img2[i][j])

cv2.imshow('Gray_difference', gray_dif_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

