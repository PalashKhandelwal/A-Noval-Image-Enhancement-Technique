import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('denoised_image.png', cv2.IMREAD_GRAYSCALE)

# Define contrast reduction factor (adjust as needed)
contrast_reduction_factor = 1.25

# Apply contrast reduction
low_intensity = np.min(image)
high_intensity = np.max(image)
contrast_reduced_image = cv2.convertScaleAbs(image, alpha=contrast_reduction_factor,
                                             beta=(1-contrast_reduction_factor)*(high_intensity+low_intensity))

# Display the original and contrast-reduced images side by side
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('HistoogramEqualized Image')
plt.imshow(contrast_reduced_image, cmap='gray')

plt.show()

# Save the contrast-reduced image to a file
cv2.imwrite('contrast_reduced_image.png', contrast_reduced_image)
