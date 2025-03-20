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

            # Process pixels in the range of 245-255
            if np.all(center_pixel >= [245, 245, 245]) and np.all(center_pixel <= [255, 255, 255]):
                neighbors = color_image[y-1:y+2, x-1:x+2]

                if np.any(np.sum(neighbors, axis=2) < 255 * 3) and np.any(np.sum(neighbors, axis=2) > 0):
                    non_extreme_neighbors = neighbors[np.sum(neighbors, axis=2) < 255 * 3]
                    
                    # Refine adaptive weighting based on local variance
                    local_variance = np.var(non_extreme_neighbors, axis=0)
                    weight = 0.5 + 0.5 * np.exp(-local_variance / 500)
                    
                    # Compute the weighted average for non-extreme neighbors
                    weighted_average = np.sum(non_extreme_neighbors * weight, axis=0) / np.sum(weight)
                    
                    # Replace the center pixel with the weighted average
                    result_image[y, x] = weighted_average.astype(np.uint8)

    return result_image

def modified_hinr_filter(input_image):
    noise_removal_output = improved_noise_removal(input_image)
    
    # Additional Step: Apply morphological opening for removing isolated bright spots
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(noise_removal_output, cv2.MORPH_OPEN, kernel)

    # Step 2: Bilateral filter for edge preservation
    bilateral_filtered = cv2.bilateralFilter(opening, d=9, sigmaColor=20, sigmaSpace=20)

    # Step 3: Edge enhancement
    edge_enhancement = cv2.subtract(opening, bilateral_filtered)

    # Step 4: Final sharpening
    final_sharpening = cv2.add(opening, edge_enhancement)

    return final_sharpening

def calculate_metrics(input_image, output_image):
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

    return mse, mae, psnr

# Load your input image
input_image = cv2.imread('15.jpg')

# Apply the median filter
median_filtered_image = median_filter(input_image, kernel_size=3)

# Apply the mean filter
mean_filtered_image = apply_mean_filter(input_image, kernel_size=3)

# Apply the bilateral filter
bilateral_filtered_image = apply_bilateral_filter(input_image, d=9, sigma_color=75, sigma_space=75)

# Apply the modified HINR filter
hinr_filtered_image = modified_hinr_filter(input_image)

# Save the output images
cv2.imwrite('Median_Filtered.jpg', median_filtered_image)
cv2.imwrite('Mean_Filtered.jpg', mean_filtered_image)
cv2.imwrite('Bilateral_Filtered.jpg', bilateral_filtered_image)
cv2.imwrite('HINR_Filtered.jpg', hinr_filtered_image)

# Display both the input and output images
cv2.imshow('Input Image', input_image)
cv2.imshow('Median Filtered Image', median_filtered_image)
cv2.imshow('Mean Filtered Image', mean_filtered_image)
cv2.imshow('Bilateral Filtered Image', bilateral_filtered_image)
cv2.imshow('HINR Filtered Image', hinr_filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Calculate evaluation metrics
mse_median, mae_median, psnr_median = calculate_metrics(input_image, median_filtered_image)
mse_mean, mae_mean, psnr_mean = calculate_metrics(input_image, mean_filtered_image)
mse_bilateral, mae_bilateral, psnr_bilateral = calculate_metrics(input_image, bilateral_filtered_image)
mse_hinr, mae_hinr, psnr_hinr = calculate_metrics(input_image, hinr_filtered_image)

# Print evaluation metrics
print("Median Filter:")
print("  Mean Squared Error (MSE):", mse_median)
print("  Mean Absolute Error (MAE):", mae_median)
print("  Peak Signal-to-Noise Ratio (PSNR):", psnr_median)

print("\nMean Filter:")
print("  Mean Squared Error (MSE):", mse_mean)
print("  Mean Absolute Error (MAE):", mae_mean)
print("  Peak Signal-to-Noise Ratio (PSNR):", psnr_mean)

print("\nBilateral Filter:")
print("  Mean Squared Error (MSE):", mse_bilateral)
print("  Mean Absolute Error (MAE):", mae_bilateral)
print("  Peak Signal-to-Noise Ratio (PSNR):", psnr_bilateral)

print("\nModified HINR Filter:")
print("  Mean Squared Error (MSE):", mse_hinr)
print("  Mean Absolute Error (MAE):", mae_hinr)
print("  Peak Signal-to-Noise Ratio (PSNR):", psnr_hinr)
