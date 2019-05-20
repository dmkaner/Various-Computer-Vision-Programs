import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

image = cv2.imread('images/object_bluescreen.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

low = 120
high = 240

edges = cv2.Canny(image, low, high)

plt.imshow(edges, cmap='gray')
plt.show()