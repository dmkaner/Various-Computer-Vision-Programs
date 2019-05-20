#!/usr/bin/env python
# coding: utf-8

# ## Image pyramids
# 
# Take a look at how downsampling with image pyramids works.
# 
# First, we'll read in an image then construct and display a few layers of an image pyramid.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2

get_ipython().run_line_magic('matplotlib', 'inline')

# Read in the image
image = cv2.imread('images/rainbow_flag.jpg')

# Change color to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)


# In[2]:


level_1 = cv2.pyrDown(image)
level_2 = cv2.pyrDown(level_1)
level_3 = cv2.pyrDown(level_2)

# Display the images
f, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(20,10))

ax1.set_title('original')
ax1.imshow(image)

ax2.imshow(level_1)
ax2.set_xlim([0, image.shape[1]])
ax2.set_ylim([image.shape[0], 0])

ax3.imshow(level_2)
ax3.set_xlim([0, image.shape[1]])
ax3.set_ylim([image.shape[0], 0])

ax4.imshow(level_3)
ax4.set_xlim([0, image.shape[1]])
ax4.set_ylim([image.shape[0], 0])


# In[ ]:




