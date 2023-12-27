import cv2
import numpy as np
import skfuzzy as fuzz  # Requires the scikit-fuzzy library
from itertools import product

# Define membership functions and fuzzy sets
def define_membership_functions(image):
    dark = fuzz.trapmf(image, [0, 0, 50, 100])
    medium = fuzz.trimf(image, [50, 100, 150])
    bright = fuzz.trapmf(image, [100, 150, 255, 255])
    return dark, medium, bright

# Fuzzification
def fuzzify(image, dark, medium, bright):
    dark_membership = dark[image]
    medium_membership = medium[image]
    bright_membership = bright[image]
    return dark_membership, medium_membership, bright_membership

# Define fuzzy filtering rules
def fuzzy_filtering(pixel, neighbors, dark, medium, bright):
    # Define fuzzy rules here to determine the pixel's new value based on its membership and neighbors' memberships
    # Implement your fuzzy logic rules

# Fuzzy defuzzification
def defuzzify(output_membership):
    # Implement defuzzification to get a crisp value
    # For example, centroid-based defuzzification
    # Return a single value based on the fuzzy memberships

def improved_noise_removal(grayscale_image):
    rows, cols = grayscale_image.shape
    output_image = np.copy(grayscale_image)
    dark, medium, bright = define_membership_functions(grayscale_image)

    for x in range(1, rows - 1):
        for y in range(1, cols - 1):
            if grayscale_image[x, y] < 255:
                continue

            neighbors = grayscale_image[x-1:x+2, y-1:y+2]
            dark_membership, medium_membership, bright_membership = fuzzify(grayscale_image[x, y], dark, medium, bright)

            # Apply fuzzy filtering rules to update pixel value
            new_value = fuzzy_filtering(grayscale_image[x, y], neighbors, dark_membership, medium_membership, bright_membership)
            output_image[x, y] = defuzzify(new_value)

    return output_image

# The rest of your code (grid search, parameter tuning, etc.) remains unchanged.
