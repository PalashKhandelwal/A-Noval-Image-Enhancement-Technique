from PIL import Image

def convert_to_grayscale(input_path, output_path):
    # Open the image file
    image = Image.open(input_path)

    # Convert the image to grayscale
    grayscale_image = image.convert('L')

    # Save the grayscale image
    grayscale_image.save(output_path)

if __name__ == "__main__":
    # Specify the input and output file paths
    input_image_path = "5.png"  # Replace with the path to your input image
    output_image_path = "output2.jpg"  # Replace with the desired output path

    # Convert the image to grayscale and save it
    convert_to_grayscale(input_image_path, output_image_path)
