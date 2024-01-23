import cv2
import numpy as np

# Load the input and output images
input_image = cv2.imread('2.png')
output_image = cv2.imread('New.png')
#57 88 30

# Ensure both images have the same dimensions
output_image = cv2.resize(output_image, (input_image.shape[1], input_image.shape[0]))

# Convert images to grayscale for MSE and MAE calculations
input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
output_gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)

# Calculate Mean Squared Error (MSE)
mse = np.mean((input_gray - output_gray) ** 2)

# Calculate Mean Absolute Error (MAE)
mae = np.mean(np.abs(input_gray - output_gray))

# Calculate Peak Signal-to-Noise Ratio (PSNR)
max_pixel_value = 255
psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Peak Signal-to-Noise Ratio (PSNR):", psnr)
