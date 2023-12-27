import cv2
import numpy as np
from itertools import product

def grayscale_conversion(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def improved_noise_removal(grayscale_image):
    rows, cols = grayscale_image.shape
    output_image = np.copy(grayscale_image)

    for x in range(1, rows - 1):
        for y in range(1, cols - 1):
            if grayscale_image[x, y] < 255:
                continue

            neighbors = grayscale_image[x-1:x+2, y-1:y+2]
            condition1 = np.sum(neighbors == 255) >= 5

            if np.any(neighbors != 255):
                if condition1:
                    weighted_average = np.nanmean(neighbors[neighbors != 255])
                else:
                    weighted_average = np.nanmean(neighbors)
                output_image[x, y] = weighted_average

    return output_image

def evaluate_algorithm(image, params):
    grayscale_image = grayscale_conversion(image)
    noise_removal_output = improved_noise_removal(grayscale_image)
    bilateral_filtered = cv2.bilateralFilter(noise_removal_output, d=params['d'], sigmaColor=params['sigmaColor'], sigmaSpace=params['sigmaSpace'])
    edge_enhanced = cv2.subtract(noise_removal_output, bilateral_filtered)
    final_output = cv2.add(noise_removal_output, edge_enhanced)
    return final_output

# Load the input image
input_image = cv2.imread('2.png')

# Define a grid of parameters to search
param_grid = {
    'd': [5, 7, 9, 11],  # Vary 'd' parameter
    'sigmaColor': [10, 30, 50],  # Vary 'sigmaColor' parameter
    'sigmaSpace': [10, 30, 50]  # Vary 'sigmaSpace' parameter
}

best_score = float('-inf')
best_params = None

# Perform grid search
for params in product(*param_grid.values()):
    params = dict(zip(param_grid.keys(), params))
    output_image = evaluate_algorithm(input_image, params)

    # Convert input image to grayscale for comparison
    grayscale_input_image = grayscale_conversion(input_image)

    # Calculate PSNR manually
    mse = np.mean((grayscale_input_image.astype(float) - output_image.astype(float)) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))

    # Track the best parameters
    if psnr > best_score:
        best_score = psnr
        best_params = params

# Use the best parameters to process the input image
output_image = evaluate_algorithm(input_image, best_params)

# Save and display the final output
cv2.imwrite('output_image.png', output_image)
cv2.imshow('Input Image', input_image)
cv2.imshow('Final Output Image', output_image)

# Load the input image and the output image
input_image = cv2.imread('input_image.png', cv2.IMREAD_GRAYSCALE)
output_image = cv2.imread('output_image.png', cv2.IMREAD_GRAYSCALE)

# Calculate MSE (Mean Squared Error)
mse = np.mean((input_image.astype(float) - output_image.astype(float))**2)

# Calculate MAE (Mean Absolute Error)
mae = np.mean(np.abs(input_image.astype(float) - output_image.astype(float)))

# Calculate PSNR (Peak Signal-to-Noise Ratio)
max_pixel_value = 255  # Assuming 8-bit images
psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))

print(f'MSE: {mse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'PSNR: {psnr:.2f}')



cv2.waitKey(0)
cv2.destroyAllWindows()
