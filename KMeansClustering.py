import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('images/object_bluescreen.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pixVals = image.reshape((-1, 3))
pixVals = np.float32(pixVals)

k=2
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
retval, labels, centers = cv2.kmeans(pixVals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))
labels_reshape = labels.reshape(image.shape[0], image.shape[1])

plt.imshow(segmented_image)

# plt.imshow(labels_reshape==0, cmap='gray')

# cluster = 0 # the first cluster

# masked_image = np.copy(image)
# masked_image[labels_reshape == cluster] = [0, 255, 0]

# plt.imshow(masked_image)

plt.show()