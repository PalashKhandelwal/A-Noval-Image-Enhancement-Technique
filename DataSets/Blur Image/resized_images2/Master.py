import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalize_histogram(hist, target_max):
    max_freq = max(hist)
    scaling_factor = target_max / max_freq
    return [int(value * scaling_factor) for value in hist]

# Read the noisy input image (replace '2.png' with the actual image file name)
noisy_image = cv2.imread('9.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate the noise histogram
noise_histogram = np.zeros(256, dtype=int)
for y in range(noisy_image.shape[0]):
    for x in range(noisy_image.shape[1]):
        pixel_value = noisy_image[y, x]
        noise_histogram[pixel_value] += 1

# Set the target maximum frequency for all histograms
target_max_frequency = 1000  # Adjust as needed

# Normalize the histogram to have the same maximum frequency
normalized_histogram = normalize_histogram(noise_histogram, target_max_frequency)

# Display the noisy image
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')

# Display the histogram
plt.subplot(2, 1, 2)
plt.bar(range(256), normalized_histogram, width=1, color='blue')
plt.title('Normalized Histogram of Noisy Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

# Show the plots
plt.tight_layout()
plt.show()
