import cv2
import numpy as np

# Load the image
image = cv2.imread('3.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the 3x3 Gaussian kernel with fixed weights
kernel = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]]) / 16

# Create an empty array for the denoised image
denoised_image = np.zeros_like(gray_image, dtype=np.uint8)

# Get the kernel size
kernel_size = kernel.shape[0]

# Iterate over the image pixels
for x in range(kernel_size // 2, gray_image.shape[0] - kernel_size // 2):
    for y in range(kernel_size // 2, gray_image.shape[1] - kernel_size // 2):
        # Extract the local neighborhood
        local_region = gray_image[x - kernel_size // 2: x + kernel_size // 2 + 1, 
                                  y - kernel_size // 2: y + kernel_size // 2 + 1]
        
        # Apply the Gaussian blur using the kernel
        smoothed_pixel = np.sum(local_region * kernel)
        
        # Store the result in the denoised image
        denoised_image[x, y] = smoothed_pixel

# Display the original and denoised images
cv2.imshow('Original Image', gray_image)
cv2.imshow('Denoised Image', denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
