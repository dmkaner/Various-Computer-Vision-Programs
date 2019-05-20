import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

image = cv2.imread('images/object_bluescreen.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel = np.array([ [0,-1,0], [-1,4,-1], [0,-1,0] ])

filteredImage = cv2.filter2D(image, -1, kernel)

retval, binaryImage = cv2.threshold(filteredImage, 20, 255, cv2.THRESH_BINARY)

plt.imshow(binaryImage, cmap='gray')
plt.show()


