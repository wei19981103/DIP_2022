import cv2
import numpy as np
import matplotlib.pyplot as plt

#decoding function
def decode32(string):
    original = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V']
    for i in range(32):
        string = string.replace(original[i], str(i) + ' ')
    string = string.split()
    return string

def plothist(image):
    hist = [0] * 256
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hist[image[i][j]] += 1
    return hist

#create blank image
img1 = np.zeros([64, 64], np.uint8)
img2 = np.zeros([64, 64], np.uint8)
img3 = np.zeros([64, 64], np.uint8)
img4 = np.zeros([64, 64], np.uint8)

images = [img1, img2, img3, img4]

#load file and store data
file1 = open('.\\JET.64', 'r')
data1 = file1.read()
file1.close()

file2 = open('.\\LIBERTY.64', 'r')
data2 = file2.read()
file2.close()

file3 = open('.\\LINCOLN.64', 'r')
data3 = file3.read()
file3.close()

file4 = open('.\\LISA.64', 'r')
data4 = file4.read()
file4.close()

#decode
data1 = decode32(data1)
data2 = decode32(data2)
data3 = decode32(data3)
data4 = decode32(data4)

datas = [data1, data2, data3, data4]

#give values to image
for num in range(len(images)):
    img = images[num]
    dt = datas[num]
    for i in range(64):
        for j in range(64):
            img[i][j] = dt[j + i * 64]

            #normalize
            img[i][j] = round((((img[i][j] - 0) * 255) / 31) + 0)

#plot images
titles = ['JET', 'LIBERTY', 'LINCOLN', 'LISA']
original_img = [img1, img2, img3, img4]

for i in range(len(titles)):
    plt.subplot(5, 4, 1 + i)
    plt.imshow(original_img[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

#plot histogram(part 1)
histograms = [plothist(img1), plothist(img2), plothist(img3), plothist(img4)]
for i in range(4):
    plt.subplot(5, 4, 5 + i)
    plt.bar(list(range(0,256)), histograms[i], width = 0.5, edgecolor = 'black')
    plt.xticks(list(range(0,256,255)))

#add constant(part 2-1)
new_image1 = np.zeros([64, 64], np.uint8)
constant1 = 100
for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        if img1[i][j] + constant1 >= 255:
            new_image1[i][j] = 255
        else:
            new_image1[i][j] = img1[i][j] + constant1

hist1 = plothist(new_image1)

#mutiply constant(part 2-2)
new_image2 = np.zeros([64, 64], np.uint8)
constant2 = 0.5
for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        new_image2[i][j] = round(img1[i][j] * constant2)

hist2 = plothist(new_image2)

#average image(part 2-3)
new_image3 = np.zeros([64, 64], np.uint8)
for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        new_image3[i][j] = (int(img1[i][j]) + int(img4[i][j]))/2

hist3 = plothist(new_image3)

#subtract previous column(part 2-4)
new_image4 = np.zeros([64, 64], np.uint8)
for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        if i == 0:
            continue
        else:
            new_image4[i][j] = int(img1[i][j]) - int(img1[i-1][j])

hist4 = plothist(new_image4)

#plot result image & histogram
titles_result = ['JET ADD 100', 'JET Mutiply by 0.5', 'AVG of JET & LISA', 'JET Subtract prv column']
result_hist = [hist1, hist2, hist3, hist4]
result_img = [new_image1, new_image2, new_image3, new_image4]

for i in range(len(titles_result)):
    plt.subplot(5, 4, 13 + i)
    plt.imshow(result_img[i], cmap='gray', vmin = 0, vmax =255)
    plt.xticks([]), plt.yticks([])
    plt.title(titles_result[i], fontsize = 9)

for i in range(len(titles_result)):
    plt.subplot(5, 4, 17 + i)
    plt.bar(list(range(0,256)), result_hist[i], width = 0.5, edgecolor = 'black')
    plt.xticks(list(range(0,256,255)))

#create window
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
