import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

image = cv2.imread('images/object_bluescreen.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurredImage = cv2.GaussianBlur(image, (35,35), 0)

plt.imshow(blurredImage, cmap='gray')
plt.show()


