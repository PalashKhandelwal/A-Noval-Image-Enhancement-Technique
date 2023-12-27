import cv2
import numpy as np

# Load the input image
input_image = cv2.imread('2.png')

# Define the size of the median filter kernel
kernel_size = 3  # Adjust this to change the filter size

# Apply the median filter to remove impulse noise
output_image = cv2.medianBlur(input_image, kernel_size)

# Display the input and output images
cv2.imshow('Input Image', input_image)
cv2.imshow('Output Image', output_image)

# Wait for a key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
