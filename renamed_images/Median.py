import cv2
import numpy as np

def median_filter(image, kernel_size):
    # Apply median filter to the input image
    filtered_image = cv2.medianBlur(image, kernel_size)
    return filtered_image

def main():
    # Load an image with impulse noise
    input_image = cv2.imread('15.jpg')

    # Set the kernel size for the median filter (should be an odd number)
    kernel_size = 3

    # Apply median filter
    filtered_image = median_filter(input_image, kernel_size)

    # Display the original and filtered images
    cv2.imshow('Original Image', input_image)
    cv2.imshow('Filtered Image (Median Filter)', filtered_image)

    # Save the filtered image
    cv2.imwrite('Output.jpg', filtered_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
