import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from skimage import img_as_float
from skimage.io import imread
from skimage.color import rgb2gray

def add_impulse_noise(image, noise_ratio):
    """
    Add impulse noise to the given image.
    """
    noisy_image = image.copy()
    num_pixels = int(noise_ratio * image.size)
    indices = np.random.randint(0, image.size, num_pixels)
    noisy_image.flat[indices] = 1 - noisy_image.flat[indices]
    return noisy_image

def wiener_filter(image, kernel_size=3, noise_var=0.01):
    """
    Apply Wiener filter to remove noise from the given image.
    """
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    noisy_image = image.copy()

    # Add noise to the image
    noisy_image = add_impulse_noise(noisy_image, noise_var)

    # Apply Wiener filter
    noisy_fft = np.fft.fft2(noisy_image)
    kernel_fft = np.fft.fft2(kernel, s=noisy_image.shape)
    wiener_filter = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + noise_var)
    restored_fft = noisy_fft * wiener_filter
    restored_image = np.abs(np.fft.ifft2(restored_fft))

    return restored_image

# Example usage:
if __name__ == "__main__":
    # Load an example image
    original_image = imread("1.png")
    if original_image.shape[-1] == 4:  # Check if the image has an alpha channel
        original_image = original_image[:, :, :3]  # Remove the alpha channel

    gray_image = rgb2gray(original_image)

    # Apply Wiener filter
    restored_image = wiener_filter(gray_image)

    # Display the results
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Noisy Image')
    plt.subplot(1, 2, 2)
    plt.imshow(restored_image, cmap='gray')
    plt.title('Restored Image with Wiener Filter')
    plt.show()
