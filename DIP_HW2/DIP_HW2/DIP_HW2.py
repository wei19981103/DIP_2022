import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

img = cv2.imread(".\\lena.jpg", 1)
height, width, channel_num = img.shape
center_x, center_y = (width / 2, height / 2)
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
def gray1(img):
    img_gray1 = np.zeros([height, width], np.uint8)
    for i in range(height):
        for j in range(width):
            color = img[i][j]
            img_gray1[i][j] = color[0] / 3 + color[1] / 3 + color[2] / 3
    
    return img_gray1

#grayscale 2
def gray2(img):
    img_gray2 =  np.zeros([height, width], np.uint8)
    for i in range(height):
        for j in range(width):
            color = img[i][j]
            img_gray2[i][j] =  round(0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0])
    
    return img_gray2

#grayscale difference
def gray_diff(img1, img2):
    gray_dif_img = np.zeros([height, width], np.uint8)
    for i in range(height):
        for j in range(width):
          gray_dif_img[i][j] = int(img1[i][j]) - int(img2[i][j])

    return gray_dif_img

#binary image
def bin_img(gray_img, threshold):
    img_bin = np.zeros([height, width], np.uint8)
    for i in range(height):
        for j in range(width):
          if gray_img[i][j] < threshold:
              img_bin[i][j] = 0
          else:
              img_bin[i][j] = 255
          
    return img_bin

#scale image
def scale_img(img, scale):
    if scale <= 1:
        img_scale = np.zeros([height, width, 3], np.uint8)
        for i in range(height):
            for j in range(width):
                scale_i = center_y + (i - center_y) * scale
                scale_j = center_x + (j - center_x) * scale
                scale_i = round(scale_i)
                scale_j = round(scale_j)
                img_scale[scale_i][scale_j] = img[i][j]
    else:
        newsize = [round(height * scale), round(width * scale)]
        img_scale = np.zeros([newsize[0], newsize[1], 3], np.uint8)
        #interpolation
        for i in range(newsize[0]):
            for j in range(newsize[1]):             
                scale_i = math.floor(i / scale)
                scale_j = math.floor(j / scale)
                img_scale[i][j] = img[scale_i][scale_j]
    return img_scale

#adjust grayscale level
def adj_graylevel(img, gray_level):
    img_new_graylevel = np.zeros([height, width], np.uint8)
    gray_ratio = (gray_level-1)/255
    for i in range(height):
        for j in range(width):
            img_new_graylevel[i][j] = round(img[i][j] * gray_ratio)

    return img_new_graylevel

#adjust brightness
def adj_brightness(img, brightness):
    img_adjsut_brightness = np.zeros([height, width], np.uint8)
    for i in range(height):
        for j in range(width):
            if img[i][j] + brightness > 255:
                img_adjsut_brightness[i][j] = 255
            elif img[i][j] + brightness < 0:
                img_adjsut_brightness[i][j] = 0
            else:
                img_adjsut_brightness[i][j] = img[i][j] + brightness
    return img_adjsut_brightness

#adjust contrast
def adj_contrast(img, contrast):
    img_adjsut_contrast = np.zeros([height, width, 3], np.uint8)
    factor = 259.0*( contrast + 255.0 ) / (255.0*( 259.0 - contrast ))
    for i in range(height):
        for j in range(width):
                img_adjsut_contrast[i][j][0] = max(0, min(255, factor * (img[i][j][0] -128) + 128))
                img_adjsut_contrast[i][j][1] = max(0, min(255, factor * (img[i][j][1] -128) + 128))
                img_adjsut_contrast[i][j][2] = max(0, min(255, factor * (img[i][j][2] -128) + 128))

    return img_adjsut_contrast
#histogram equalization
def hist_equ(img):
    img_equ = np.zeros([height, width], np.uint8)
    new_pixelvalue = []
    for i in range(256):
        new_pixelvalue.append([i])
    cdf = 0
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
cv2.imshow('Gray_difference', img_gray_diff)

hist_gray1 = draw_hist(img_gray1)
hist_gray2 = draw_hist(img_gray2)

#plt.bar(list(range(0,256)), hist_gray1, width = 0.5, edgecolor = 'black')
#plt.xticks(list(range(0,256,50)))

#plt.bar(list(range(0,256)), hist_gray2, width = 0.5, edgecolor = 'black')
#plt.xticks(list(range(0,256,50)))

#plt.show()

#binary imgae
threshold = 127
img_bin = bin_img(img_gray2, threshold)
cv2.imshow('Binary imgae', img_bin)

#scale image
scale = 2
img_scale = scale_img(img, scale)
cv2.imshow('Scale image', img_scale)

#adjust grayscale
grayscale_level = 2
img_new_graylevel = adj_graylevel(img_gray1, grayscale_level)
plt.imshow(img_new_graylevel, cmap='gray', vmin = 0, vmax = grayscale_level-1)
#plt.show()

#adjust brightness
brightness = 50
img_adjust_brightness = adj_brightness(img_gray1, brightness)
cv2.imshow('Adjust brightness', img_adjust_brightness)

#adjust contrast
contrast = 127
img_adjust_contrast = adj_contrast(img, contrast)
cv2.imshow('Adjust contrast', img_adjust_contrast)

#histogram equalization
img_equ = hist_equ(img_gray2)
cv2.imshow('Histogram equalization', img_equ)

cv2.waitKey(0)
cv2.destroyAllWindows()

