import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(".\\lena.jpg", 1)
height, width, channel_num = img.shape
cv2.imshow('Original',img)

#draw Histogram
def draw_hist(img):
    hist = [0] * 256
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            hist[img[i][j]] += 1
    return hist

#grayscale 1
img_gray1 = np.zeros([height, width], np.uint8)
def gray1(img):
    for i in range(height):
        for j in range(width):
            color = img[i][j]
            img_gray1[i][j] = color[0] / 3 + color[1] / 3 + color[2] / 3
    
    return img_gray1

#grayscale 2
img_gray2 =  np.zeros([height, width], np.uint8)
def gray2(img):
    for i in range(height):
        for j in range(width):
            color = img[i][j]
            img_gray2[i][j] =  round(0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0])
    
    return img_gray2

#grayscale difference
gray_dif_img = np.zeros([height, width], np.uint8)
def gray_diff(img1, img2):
    for i in range(height):
        for j in range(width):
          gray_dif_img[i][j] = int(img1[i][j]) - int(img2[i][j])

    return gray_dif_img

#binary image
img_bin = np.zeros([height, width], np.uint8)
def bin_img(img, threshold):
    for i in range(height):
        for j in range(width):
          if img[i][j] < threshold:
              img_bin[i][j] = 0
          else:
              img_bin[i][j] = 255
          
    return img_bin

#histogram equalization
img_equ = np.zeros([height, width], np.uint8)
new_pixelvalue = []
for i in range(256):
    new_pixelvalue.append([i])
cdf = 0
def hist_equ(img):
    hist = draw_hist(img)
    for k in range(len(hist)):
        for j in range(k+1):
            cdf += hist[j] / (height * width)
        new_pixelvalue[k] = round(255 * cdf) #list[original_pixelvalue] = new_pixelvalue
        cdf = 0
    for i in range(height):
        for j in range(width):
            img_equ[i][j] = new_pixelvalue[img[i][j]]

    return img_equ

########### main ###########

#gray 1
img_gray1 = gray1(img)
cv2.imshow('Gray1', img_gray1)

#gray 2
img_gray2 = gray2(img)
cv2.imshow('Gray2', img_gray2)

#gray difference
img_gray_diff = gray_diff(img_gray1, img_gray2)
#cv2.imshow('Gray_difference', img_gray_diff)

hist_gray1 = draw_hist(img_gray1)
hist_gray2 = draw_hist(img_gray2)

#plt.bar(list(range(0,256)), hist_gray1, width = 0.5, edgecolor = 'black')
#plt.xticks(list(range(0,256,50)))

#plt.bar(list(range(0,256)), hist_gray2, width = 0.5, edgecolor = 'black')
#plt.xticks(list(range(0,256,50)))

#plt.show()

#binary imgae
threshold = 92
img_bin = bin_img(img_gray2, threshold)
cv2.imshow('binary imgae', img_bin)

#histogram equalization

cv2.waitKey(0)
cv2.destroyAllWindows()

