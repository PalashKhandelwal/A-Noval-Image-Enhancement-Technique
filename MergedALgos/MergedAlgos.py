import cv2
import numpy as np

def apply_bilateral_filter(input_image, d=9, sigma_color=75, sigma_space=75):
    # Apply bilateral filter to the input image
    filtered_image = cv2.bilateralFilter(input_image, d, sigma_color, sigma_space)
    return filtered_image

def apply_mean_filter(input_image, kernel_size=3):
    # Apply mean filter to the input image
    filtered_image = cv2.blur(input_image, (kernel_size, kernel_size))
    return filtered_image

def median_filter(image, kernel_size):
    # Apply median filter to the input image
    filtered_image = cv2.medianBlur(image, kernel_size)
    return filtered_image

def improved_noise_removal(color_image):
    height, width, channels = color_image.shape
    result_image = color_image.copy()

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            center_pixel = color_image[y, x]

            if np.all(center_pixel == [255, 255, 255]):  # Process only white pixels
                neighbors = color_image[y-1:y+2, x-1:x+2]

                if np.any(np.sum(neighbors, axis=2) != 255 * 3):
                    non_extreme_neighbors = neighbors[np.sum(neighbors, axis=2) != 255 * 3]
                    
                    # Refine adaptive weighting based on local variance
                    local_variance = np.var(non_extreme_neighbors, axis=0)
                    weight = 0.5 + 0.5 * np.exp(-local_variance / 500)
                    
                    weighted_average = np.sum(non_extreme_neighbors * weight, axis=0) / np.sum(weight)
                    result_image[y, x] = weighted_average.astype(np.uint8)

    return result_image

def modified_hinr_filter(input_image):
    noise_removal_output = improved_noise_removal(input_image)
    
    # Step 2: Bilateral filter for edge preservation
    bilateral_filtered = cv2.bilateralFilter(noise_removal_output, d=9, sigmaColor=20, sigmaSpace=20)

    # Step 3: Edge enhancement
    edge_enhancement = cv2.subtract(noise_removal_output, bilateral_filtered)

    # Step 4: Final sharpening
    final_sharpening = cv2.add(noise_removal_output, edge_enhancement)

    return final_sharpening

# Load your input image
input_image = cv2.imread('5.png')

# Apply bilateral filter
filtered_image_bilateral = apply_bilateral_filter(input_image)

# Apply mean filter
filtered_image_mean = apply_mean_filter(input_image)

# Apply median filter
filtered_image_median = median_filter(input_image, kernel_size=3)

# Apply modified HINR filter
filtered_image_hinr = modified_hinr_filter(input_image)

# Display the input and output images for each filter without resizing
cv2.imshow('Input Image', input_image)
cv2.imshow('Bilateral Filter', filtered_image_bilateral)
cv2.imshow('Mean Filter', filtered_image_mean)
cv2.imshow('Median Filter', filtered_image_median)
cv2.imshow('Modified HINR Filter', filtered_image_hinr)

# Save the original output images
cv2.imwrite('Bilateral.png', filtered_image_bilateral)
cv2.imwrite('Mean.png', filtered_image_mean)
cv2.imwrite('Median.png', filtered_image_median)
cv2.imwrite('HINR.png', filtered_image_hinr)

# Load the output images for comparison
output_bilateral = cv2.imread('Bilateral.png')
output_mean = cv2.imread('Mean.png')
output_median = cv2.imread('Median.png')
output_hinr = cv2.imread('HINR.png')

# Ensure both images have the same dimensions
output_bilateral = cv2.resize(output_bilateral, (input_image.shape[1], input_image.shape[0]))
output_mean = cv2.resize(output_mean, (input_image.shape[1], input_image.shape[0]))
output_median = cv2.resize(output_median, (input_image.shape[1], input_image.shape[0]))
output_hinr = cv2.resize(output_hinr, (input_image.shape[1], input_image.shape[0]))

# Convert images to grayscale for MSE and MAE calculations
input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
output_gray_bilateral = cv2.cvtColor(output_bilateral, cv2.COLOR_BGR2GRAY)
output_gray_mean = cv2.cvtColor(output_mean, cv2.COLOR_BGR2GRAY)
output_gray_median = cv2.cvtColor(output_median, cv2.COLOR_BGR2GRAY)
output_gray_hinr = cv2.cvtColor(output_hinr, cv2.COLOR_BGR2GRAY)

# Calculate Mean Squared Error (MSE) for each filter
mse_bilateral = np.mean((input_gray - output_gray_bilateral) ** 2)
mse_mean = np.mean((input_gray - output_gray_mean) ** 2)
mse_median = np.mean((input_gray - output_gray_median) ** 2)
mse_hinr = np.mean((input_gray - output_gray_hinr) ** 2)

# Calculate Mean Absolute Error (MAE) for each filter
mae_bilateral = np.mean(np.abs(input_gray - output_gray_bilateral))
mae_mean = np.mean(np.abs(input_gray - output_gray_mean))
mae_median = np.mean(np.abs(input_gray - output_gray_median))
mae_hinr = np.mean(np.abs(input_gray - output_gray_hinr))

# Calculate Peak Signal-to-Noise Ratio (PSNR) for each filter
max_pixel_value = 255
psnr_bilateral = 20 * np.log10(max_pixel_value / np.sqrt(mse_bilateral))
psnr_mean = 20 * np.log10(max_pixel_value / np.sqrt(mse_mean))
psnr_median = 20 * np.log10(max_pixel_value / np.sqrt(mse_median))
psnr_hinr = 20 * np.log10(max_pixel_value / np.sqrt(mse_hinr))

# Display metrics for each filter
print("Bilateral Filter:")
print("  Mean Squared Error (MSE):", mse_bilateral)
print("  Mean Absolute Error (MAE):", mae_bilateral)
print("  Peak Signal-to-Noise Ratio (PSNR):", psnr_bilateral)

print("\nMean Filter:")
print("  Mean Squared Error (MSE):", mse_mean)
print("  Mean Absolute Error (MAE):", mae_mean)
print("  Peak Signal-to-Noise Ratio (PSNR):", psnr_mean)

print("\nMedian Filter:")
print("  Mean Squared Error (MSE):", mse_median)
print("  Mean Absolute Error (MAE):", mae_median)
print("  Peak Signal-to-Noise Ratio (PSNR):", psnr_median)

print("\nModified HINR Filter:")
print("  Mean Squared Error (MSE):", mse_hinr)
print("  Mean Absolute Error (MAE):", mae_hinr)
print("  Peak Signal-to-Noise Ratio (PSNR):", psnr_hinr)

# Wait for user input and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
