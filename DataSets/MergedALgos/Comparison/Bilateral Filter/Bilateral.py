import cv2

def apply_bilateral_filter(input_image, d=9, sigma_color=75, sigma_space=75):
    # Apply bilateral filter to the input image
    filtered_image = cv2.bilateralFilter(input_image, d, sigma_color, sigma_space)
    return filtered_image

# Load your input image
input_image = cv2.imread('5.png')

# Apply bilateral filter
filtered_image = apply_bilateral_filter(input_image)

# Display the input and output images
cv2.imshow('Input Image', input_image)
cv2.imshow('Filtered Image', filtered_image)

# Save the denoised image
cv2.imwrite('Output.png', filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
