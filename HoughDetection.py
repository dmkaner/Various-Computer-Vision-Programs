import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

image = cv2.imread('images/object_bluescreen.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

low = 120
high = 240

rho = 1
theta = np.pi/100
threshold = 60
minLineLength = 50
maxLineGap = 5

edges = cv2.Canny(image, low, high)
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength, maxLineGap)

for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 5)

plt.imshow(image, cmap='gray')
plt.show()