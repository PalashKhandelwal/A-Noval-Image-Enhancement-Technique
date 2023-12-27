import cv2
import numpy as np
import random

# Load the image
image = cv2.imread('your_image.jpg')  # Replace 'your_image.jpg' with the path to your image file

# Define the noise probability (e.g., 0.01 for 1% noise)
noise_probability = 0.01

# Create a copy of the image to add noise to
noisy_image = np.copy(image)

# Get the dimensions of the image
height, width, channels = image.shape

# Loop through each pixel in the image
for y in range(height):
    for x in range(width):
        # Randomly decide whether to add noise
        if random.random() < noise_probability:
            # Generate a random intensity value for the noise (0 for black, 255 for white)
            noise_intensity = random.choice([0, 255])
            # Add the noise to each channel (for color images)
            for channel in range(channels):
                noisy_image[y, x, channel] = noise_intensity

# Define the path where you want to save the noisy image
output_path = 'path_to_desired_location/noisy_image.jpg'  # Replace with your desired location and file name

# Save the noisy image to the specified location
cv2.imwrite(output_path, noisy_image)

# Display the noisy image (optional)
cv2.imshow('Noisy Image', noisy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
