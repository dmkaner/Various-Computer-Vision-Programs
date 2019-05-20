import matplotlib.pyplot as plt
import numpy as np 
import cv2

image = cv2.imread('images/object_bluescreen.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

r = image[:, :, 0]
g = image[:, :, 1]
b = image[:, :, 2]

hsvImage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

h = hsvImage[:, :, 0]
s = hsvImage[:, :, 1]
v = hsvImage[:, :, 2]

lowerHue = np.array([160,0,0])
upperHue = np.array([180,255,255])

maskHSV = cv2.inRange(hsvImage, lowerHue, upperHue)
image[maskHSV==0] = [0,0,0]

plt.imshow(image)
plt.show()