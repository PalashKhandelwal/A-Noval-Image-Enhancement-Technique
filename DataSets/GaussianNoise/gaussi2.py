import cv2
import numpy as np

# Load the image
image = cv2.imread('4.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the size of the local neighborhood
neighborhood_size = 7

# Create an empty array for the denoised image
denoised_image = np.zeros_like(gray_image, dtype=np.uint8)

# Get the kernel size
kernel_size = 3

# Iterate over the image pixels
for x in range(kernel_size // 2, gray_image.shape[0] - kernel_size // 2):
    for y in range(kernel_size // 2, gray_image.shape[1] - kernel_size // 2):
        # Define the local neighborhood
        local_region = gray_image[x - neighborhood_size // 2: x + neighborhood_size // 2 + 1,
                                  y - neighborhood_size // 2: y + neighborhood_size // 2 + 1]

        # Compute the adaptive standard deviation for the local neighborhood
        local_std = np.std(local_region)

        # Calculate the weights for the kernel based on the local standard deviation
        kernel_weights = np.exp(-0.5 * (local_region - gray_image[x, y]) ** 2 / local_std ** 2)
        kernel_weights /= np.sum(kernel_weights)

        # Apply the Gaussian blur using the adaptive weights
        smoothed_pixel = np.sum(local_region * kernel_weights)

        # Store the result in the denoised image
        denoised_image[x, y] = smoothed_pixel

# Apply sharpening using a sharpening filter
sharpening_filter = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
sharpened_image = cv2.filter2D(denoised_image, -1, sharpening_filter)

# Display the original, denoised, and sharpened images
cv2.imshow('Original Image', gray_image)
cv2.imshow('Denoised Image', denoised_image)
cv2.imshow('Sharpened Image', sharpened_image)

# Save the denoised and sharpened images to files
cv2.imwrite('denoised_image.png', denoised_image)
cv2.imwrite('sharpened_image.png', sharpened_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
