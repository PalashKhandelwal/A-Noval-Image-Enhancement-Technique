import cv2
import numpy as np

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
                    
                    # Compute the weighted average for non-extreme neighbors
                    weighted_average = np.sum(non_extreme_neighbors * weight, axis=0) / np.sum(weight)
                    
                    # Replace the center pixel with the weighted average
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
input_image = cv2.imread('1.png')

# Apply the modified HINR filter
filtered_image = modified_hinr_filter(input_image)

# Save the output image
cv2.imwrite('HINR.png', filtered_image)

# Display both the input and output images
cv2.imshow('Input Image', input_image)
cv2.imshow('Output Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
