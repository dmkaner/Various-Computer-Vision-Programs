import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

imageOriginal = cv2.imread('images/cornerImage.png')
imageOriginal = cv2.cvtColor(imageOriginal, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(imageOriginal, cv2.COLOR_RGB2GRAY)

image = np.float32(image)

dst = cv2.cornerHarris(image, 2, 3, 0.04)
dst = cv2.dilate(dst, None)

threshold = 0.01*dst.max()

for i in range(0, dst.shape[0]):
    for e in range(0, dst.shape[1]):
        if dst[i][e] > threshold:
            cv2.circle(imageOriginal, (e, i), 1, (0,255,0), 1)

plt.imshow(imageOriginal)
plt.show()


# dilation and erosion for open and closed noise reduction

# Reads in a binary image
# image = cv2.imread(‘j.png’, 0) \
# kernel = np.ones((5,5),np.uint8)\
# dilation = cv2.dilate(image, kernel, iterations = 1)
# erosion = cv2.erode(image, kernel, iterations = 1)

# opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
