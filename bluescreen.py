import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('images/object_bluescreen.jpg')
print('This image is:', type(image), 'with dimensions:', image.shape)

imageCopy = np.copy(image)
imageCopy = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2RGB)

# plt.imshow(imageCopy)
# plt.show()

lowerBlue = np.array([0, 0, 220])
upperBlue = np.array([70, 70, 255])

mask = cv2.inRange(imageCopy, lowerBlue, upperBlue)

# plt.imshow(mask, cmap='gray')
# plt.show()

maskedImage = np.copy(imageCopy)
maskedImage[mask != 0] = [0, 0, 0]

# plt.imshow(maskedImage)
# plt.show()

imageBG = cv2.imread('images/spacebg.jpg')
imageBG = cv2.cvtColor(imageBG, cv2.COLOR_BGR2RGB)
croppedImageBG = imageBG[0:720, 0:1280]
croppedImageBG[mask == 0] = [0, 0, 0]

completeImage = maskedImage + croppedImageBG
plt.imshow(completeImage)
plt.show()
