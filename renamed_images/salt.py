import cv2
import numpy as np

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)

    # Add salt noise
    salt_mask = np.random.rand(*image.shape[:2]) < salt_prob
    noisy_image[salt_mask] = 255  # Set salt pixels to 255 (white)

    # Add pepper noise
    pepper_mask = np.random.rand(*image.shape[:2]) < pepper_prob
    noisy_image[pepper_mask] = 0  # Set pepper pixels to 0 (black)

    return noisy_image

# Load an image
image_path = "1.jpg"
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define probability for salt and pepper noise
salt_probability = 0.02  # Adjust this value according to your preference
pepper_probability = 0.02  # Adjust this value according to your preference

# Add salt and pepper noise
noisy_image = add_salt_and_pepper_noise(original_image, salt_probability, pepper_probability)

# Display the original and noisy images
cv2.imshow("Original Image", original_image)
cv2.imshow("Noisy Image", noisy_image)

# Save the noisy image
cv2.imwrite('Salt.jpg', noisy_image)

# Wait for a key event and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
