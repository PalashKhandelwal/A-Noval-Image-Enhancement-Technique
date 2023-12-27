'''
    Mean Squared Error (MSE):
        Input Image MSE: 8.857
        Output Image MSE: 33.324

    The output image has a higher MSE compared to the input image. 
    This indicates that, after applying the impulse noise removal algorithm, there is a larger difference between the processed image and the original image in terms of pixel values. 
    A higher MSE suggests more significant changes to the image.

    Mean Absolute Error (MAE):
        Input Image MAE: 10.737
        Output Image MAE: 95.501

    Similar to MSE, the output image has a significantly higher MAE compared to the input image. 
    This indicates a larger difference in pixel intensity between the processed image and the original image.

    Peak Signal-to-Noise Ratio (PSNR):
        Input Image PSNR: 15.812
        Output Image PSNR: 17.243

    The output image has a slightly higher PSNR compared to the input image, indicating a slightly better signal-to-noise ratio. 
    While the PSNR of the output image is higher, the difference is not substantial.

In conclusion, after applying the impulse noise removal algorithm, the processed output image has a higher MSE and MAE compared to the original image. This suggests that the algorithm has introduced changes to the image, which are reflected in higher error metrics. 
However, the small improvement in PSNR suggests a slightly better signal-to-noise ratio in the output image.

'''
