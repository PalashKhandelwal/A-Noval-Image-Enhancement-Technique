import cv2
import numpy as np

# Step 1: Grayscale Conversion
def grayscale_conversion(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale

# Step 2: Improved Noise Removal
def improved_noise_removal(grayscale_image):
    rows, cols = grayscale_image.shape
    output_image = np.copy(grayscale_image)
    
    for x in range(1, rows - 1):
        for y in range(1, cols - 1):
            if grayscale_image[x, y] < 255:
                continue
            
            neighbors = grayscale_image[x-1:x+2, y-1:y+2]
            condition1 = np.sum(neighbors == 255) >= 5
            
            if condition1:
                weighted_average = np.mean(neighbors[neighbors != 255])
            else:
                weighted_average = np.mean(neighbors)
            
            output_image[x, y] = weighted_average
    
    return output_image

# Step 3: Bilateral Filter for Edge Preservation
def bilateral_filter(image):
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

# Step 4: Edge Enhancement
def edge_enhancement(image1, image2):
    return cv2.subtract(image1, image2)

# Step 5: Final Sharpening
def final_sharpening(image1, image2):
    return cv2.add(image1, image2)

# Load the input image
input_image = cv2.imread('1.png')

# Apply the steps of the algorithm
grayscale_image = grayscale_conversion(input_image)
noise_removal_output = improved_noise_removal(grayscale_image)
bilateral_filtered = bilateral_filter(noise_removal_output)
edge_enhanced = edge_enhancement(noise_removal_output, bilateral_filtered)
final_output = final_sharpening(noise_removal_output, edge_enhanced)

# Save the final output
cv2.imwrite('output_image.png', final_output)
