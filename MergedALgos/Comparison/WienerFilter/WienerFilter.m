% Read the input image
original_image = imread('C:/Users/Palash/Desktop/M.C.A/Thesis/DataSets/ImpulseNoiseImages/Comparison/WienerFilter/5.png'); % Replace with the actual image path

% Convert the image to double precision
original_image = im2double(original_image);

% Check if the image is truecolor and convert to grayscale
if size(original_image, 3) == 3
    original_image_gray = rgb2gray(original_image);
else
    original_image_gray = original_image;
end

% Set the Wiener filter parameters
window_size = 3; % Adjust as needed

% Apply Wiener filter
restored_image = wiener2(original_image_gray, [window_size, window_size]);

% Display the results
figure;
subplot(1, 2, 1), imshow(original_image), title('Original Image');
subplot(1, 2, 2), imshow(restored_image), title('Restored Image with Wiener Filter');

% Save the restored image
output_path = 'C:/Users/Palash/Desktop/M.C.A/Thesis/DataSets/ImpulseNoiseImages/Comparison/WienerFilter/restored_image.png'; % Replace with the desired output path
imwrite(restored_image, output_path);
