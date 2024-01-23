import cv2
import numpy as np

def grayscale_conversion(image):
    # Convert RGB image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def improved_noise_removal(grayscale_image):
    height, width = grayscale_image.shape
    result_image = grayscale_image.copy()

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            center_pixel = grayscale_image[y, x]

            if center_pixel == 255:  # Process only white pixels
                neighbors = grayscale_image[y-1:y+2, x-1:x+2]

                if np.any(neighbors != 255):
                    non_extreme_neighbors = neighbors[neighbors != 255]
                    
                    # Refine adaptive weighting based on local variance
                    local_variance = np.var(neighbors)
                    weight = 0.5 + 0.5 * np.exp(-local_variance / 500)
                    
                    weighted_average = np.sum(non_extreme_neighbors * weight) / np.sum(weight)
                    result_image[y, x] = weighted_average

    return result_image

def modified_hinr_filter(input_image):
    grayscale_image = grayscale_conversion(input_image)
    noise_removal_output = improved_noise_removal(grayscale_image)
    
    # Fine-tune bilateral filter parameters
    bilateral_filtered = cv2.bilateralFilter(noise_removal_output, d=9, sigmaColor=20, sigmaSpace=20)
    edge_enhancement = cv2.subtract(noise_removal_output, bilateral_filtered)
    final_sharpening = cv2.add(noise_removal_output, edge_enhancement)

    return final_sharpening

# Load your input image
input_image = cv2.imread('5.png')

# Apply the modified HINR filter
filtered_image = modified_hinr_filter(input_image)

# Save the output image
cv2.imwrite('output_image.jpg', filtered_image)

# Display both the input and output images
cv2.imshow('Input Image', input_image)
cv2.imshow('Output Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
