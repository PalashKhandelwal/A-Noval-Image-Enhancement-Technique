import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the noisy input image (replace 'your_noisy_image.jpg' with the actual image file name)
noisy_image = cv2.imread('5.png', cv2.IMREAD_GRAYSCALE)

# Display the noisy image
plt.figure(figsize=(8, 4))
plt.subplot(2, 1, 1)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')

# Create a histogram of the noisy image
plt.subplot(2, 1, 2)
plt.hist(noisy_image.ravel(), bins=256, range=[0, 256], histtype='step', color='black')
plt.title('Histogram of Noisy Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

# Show the plots
plt.tight_layout()
plt.show()
