import cv2
import os

# Create a directory to store the resized images
if not os.path.exists('resized_images2'):
    os.makedirs('resized_images2')

# List all image files in the current directory
image_files = [f for f in os.listdir() if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'))]

# Define the new dimensions (256x256)
new_width = 400
new_height = 400

# Loop through each image file and resize it
for image_file in image_files:
    # Load the image
    image = cv2.imread(image_file)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    # Save the resized image to the 'resized_images' directory
    output_path = os.path.join('resized_images2', image_file)
    cv2.imwrite(output_path, resized_image)

print("All images resized and saved in 'resized_images2' directory.")
