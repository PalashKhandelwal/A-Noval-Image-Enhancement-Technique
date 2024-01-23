import cv2
import numpy as np

def improved_noise_removal(color_image, threshold=500):
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
                    weight = 0.5 + 0.5 * np.exp(-local_variance / threshold)
                    
                    weighted_average = np.sum(non_extreme_neighbors * weight, axis=0) / np.sum(weight)
                    result_image[y, x] = weighted_average.astype(np.uint8)

    return result_image

def modified_hinr_filter(input_image, bilateral_d=9, bilateral_sigma_color=20, bilateral_sigma_space=20):
    noise_removal_output = improved_noise_removal(input_image)
    
    # Step 2: Bilateral filter for edge preservation
    bilateral_filtered = cv2.bilateralFilter(noise_removal_output, d=bilateral_d, sigmaColor=bilateral_sigma_color, sigmaSpace=bilateral_sigma_space)

    # Step 3: Edge enhancement
    edge_enhancement = cv2.subtract(noise_removal_output, bilateral_filtered)

    # Combine noise removal output and edge enhancement
    final_output = cv2.add(noise_removal_output, edge_enhancement)

    return final_output

# Load your input image
input_image = cv2.imread('2.png')

# Apply the modified HINR filter with adjusted parameters
filtered_image = modified_hinr_filter(input_image, bilateral_d=9, bilateral_sigma_color=20, bilateral_sigma_space=20)

# Save the output image
cv2.imwrite('New.png', filtered_image)

# Display both the input and output images
cv2.imshow('Input Image', input_image)
cv2.imshow('Output Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
