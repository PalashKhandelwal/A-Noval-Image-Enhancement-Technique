import cv2
import numpy as np

def apply_mean_filter(input_image, kernel_size=3):
    # Apply mean filter to the input image
    filtered_image = cv2.blur(input_image, (kernel_size, kernel_size))
    return filtered_image

# Load your input image
input_image = cv2.imread('5.png')

# Apply mean filter
filtered_image = apply_mean_filter(input_image)

# Display the input and output images
cv2.imshow('Input Image', input_image)
cv2.imshow('Filtered Image', filtered_image)

# Save the denoised image
cv2.imwrite('Output.png', filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
