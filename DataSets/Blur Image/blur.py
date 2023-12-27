import numpy as np
import cv2
from scipy.signal import convolve2d

def wiener_deconvolution(blurred_image, kernel, noise_variance):
    # Fourier transform of the input data
    blurred_fft = np.fft.fft2(blurred_image)
    kernel_fft = np.fft.fft2(kernel, s=blurred_image.shape)
    
    # Wiener deconvolution formula
    wiener_filter = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + noise_variance)
    restored_fft = blurred_fft * wiener_filter
    
    # Inverse Fourier transform to get the restored image
    restored_image = np.abs(np.fft.ifft2(restored_fft)).astype(np.uint8)
    
    return restored_image

# Example usage
if __name__ == "__main__":
    # Load a blurred image
    blurred_image = cv2.imread('6.jpg', cv2.IMREAD_GRAYSCALE)
    
    # Define a blur kernel (for example, a 3x3 motion blur kernel)
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2
    
    # Add simulated noise (you may need to adjust the noise level based on your specific case)
    noise_variance = 1e-3
    blurred_image += np.random.normal(scale=np.sqrt(noise_variance), size=blurred_image.shape).astype(np.uint8)
    
    # Perform Wiener deconvolution
    restored_image = wiener_deconvolution(blurred_image, kernel, noise_variance)
    
    # Display the results
    cv2.imshow('Blurred Image', blurred_image)
    cv2.imshow('Restored Image (Wiener Deconvolution)', restored_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
