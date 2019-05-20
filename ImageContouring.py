import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

imageOriginal = cv2.imread('images/hand.jpg')
imageOriginal = cv2.cvtColor(imageOriginal, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(imageOriginal, cv2.COLOR_RGB2GRAY)

retval, binary = cv2.threshold(image, 225, 255, cv2.THRESH_BINARY_INV)

retval, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

allContours = cv2.drawContours(imageOriginal, contours, -1, (0,255,0), 2)

plt.imshow(allContours, cmap='gray')
plt.show()