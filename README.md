# Image to Pixel Conversion

This project demonstrates a Python script that converts images into their pixel representations using direct extraction methods. It utilizes popular libraries such as Pillow, NumPy, Matplotlib, and Pandas to handle image processing and data representation.

## Approach

The script takes an image as input, converts it to RGB format, and extracts pixel data. The pixel values are stored in a 3D NumPy array, which is then saved to a file for further analysis. Additionally, the pixel data can be visualized and converted to a Pandas DataFrame for easier manipulation.

## Libraries Used

- **Pillow**: For image handling and processing.
- **NumPy**: For numerical operations and array handling.
- **Matplotlib**: For visualizing images and pixel data.
- **Pandas**: For converting pixel data into a DataFrame format.

## Output Format

The output pixel representation is saved as a NumPy binary file (`.npy`), containing a 3D array where:
- Each element represents a pixel's RGB values.
- The shape of the array is `(height, width, 3)`.

## Usage

1. Run the script and upload an image when prompted.
2. The script will convert the image to pixel data and save it as `pixel_data.npy`.
3. It will also display the original image and print some pixel information.
4. Finally, you can download the pixel data file for further use.

## Example

After running the script with an example image, you will see the pixel data saved, the original image displayed, and a sample of the pixel values printed in the console.

# Here is the Python Code:

# Direct-Pixel-Extraction

Install necessary libraries
"""

# Install necessary libraries
!pip install pillow matplotlib pandas

"""Import the required libraries"""

# Import the required libraries
from PIL import Image                 # For image processing
import numpy as np                    # For numerical operations with arrays
import matplotlib.pyplot as plt       # For visualizing images
import pandas as pd                   # For handling data in DataFrame format
from google.colab import files        # For file upload and download in Google Colab

"""Define necessary functions required to perform Direct Pixel Extraction"""

"""
    Function image_to_pixels:

    To convert an image to its pixel representation.

    Args:
    image_path (str): Path to the input image file.

    Returns:
    numpy.ndarray: A 3D numpy array containing the RGB values of each pixel.
"""
def image_to_pixels(image_path):

    # Open the image file using the PIL library
    with Image.open(image_path) as img:
        # Convert the image to RGB mode if it is not already in RGB
        img = img.convert('RGB')

        # Convert the image to a numpy array to facilitate pixel manipulation
        pixel_data = np.array(img)

    return pixel_data  # Return the pixel data as a 3D numpy array


"""
    Function save_pixel_data:

    To save the pixel data to a file.

    Args:
    pixel_data (numpy.ndarray): The pixel data to be saved.
    output_file (str): Path to the output file.
"""
def save_pixel_data(pixel_data, output_file):

    # Save the pixel data as a numpy file in .npy format
    np.save(output_file, pixel_data)


"""
    Function display_image_from_pixels:

    To display an image from its pixel data.

    Args:
    pixel_data (numpy.ndarray): The pixel data of the image.
"""
def display_image_from_pixels(pixel_data):

    # Use Matplotlib to display the image from the pixel data
    plt.imshow(pixel_data)  # Show the image
    plt.axis('off')  # Turn off the axis labels
    plt.show()  # Render the image on the screen


"""
    Function visualize_pixel_info:

    To visualize the extracted pixel information.

    Args:
    pixel_data (numpy.ndarray): The pixel data of the image.
"""
def visualize_pixel_info(pixel_data):

    # Create a figure to display the image
    plt.figure(figsize=(10, 10))  # Set the figure size
    plt.imshow(pixel_data)  # Show the image
    plt.title("Original Image")  # Set the title for the plot
    plt.axis('off')  # Turn off the axis labels
    plt.show()  # Render the image on the screen

    # Extract the dimensions of the pixel data
    height, width, _ = pixel_data.shape
    # Print the dimensions of the image
    print(f"Image Dimensions: {height} x {width}")

    # Display some sample pixel values from the top-left corner of the image
    sample_pixels = pixel_data[:10, :10, :]  # Get the first 10x10 pixels
    print("Sample pixel values (first 10x10 pixels):")
    print(sample_pixels)  # Print the sample pixel values

"""
    Function pixel_data_to_dataframe:

    To convert pixel data to a pandas DataFrame.

    Args:
    pixel_data (numpy.ndarray): The pixel data of the image.

    Returns:
    pd.DataFrame: A DataFrame containing the pixel data.
"""
def pixel_data_to_dataframe(pixel_data):

    # Reshape the pixel data to a 2D array where each row represents a pixel
    pixels_reshaped = pixel_data.reshape(-1, pixel_data.shape[-1])

    # Create a DataFrame with columns for R, G, B values
    df = pd.DataFrame(pixels_reshaped, columns=['R', 'G', 'B'])
    return df  # Return the DataFrame containing pixel data

"""Upload the image"""

# Upload the image from the local machine
uploaded = files.upload()

# Get the uploaded image file name
input_image_path = list(uploaded.keys())[0]  # Extract the first (and only) file name
output_pixel_data_file = 'pixel_data.npy'  # Specify the output file name for pixel data

"""Image to pixel Conversion"""

# Convert the uploaded image to pixel data
pixels = image_to_pixels(input_image_path)

"""Saving the pixel data to a Numpy file"""

# Save the extracted pixel data to a file
save_pixel_data(pixels, output_pixel_data_file)

# Notify the user that pixel data has been saved
print(f"Pixel data saved to {output_pixel_data_file}")

"""Displaying the Numpy file as image"""

# Display the pixel data as an image
display_image_from_pixels(pixels)

# Visualize the extracted pixel information
visualize_pixel_info(pixels)

"""Displaying the pixels as Dataframe"""

# Convert pixel data to a DataFrame and display it
pixel_df = pixel_data_to_dataframe(pixels)  # Convert to DataFrame
print("Pixel DataFrame:")  # Print header for DataFrame output
print(pixel_df.head(10))  # Display the first 10 rows of the DataFrame
print(pixel_df)            # Display the complete DataFrame

# Download the pixel data file to the local machine
files.download(output_pixel_data_file)

# Load the pixel data from the saved file
pixel_data = np.load('pixel_data.npy')

# Print the shape of the pixel data (height, width, 3)
print(pixel_data.shape)

"""Dipslay the pixel data as a NumPy array"""

# Print the pixel data array
print(pixel_data)

"""Dipslay the pixel data as a list"""

pixel_data.tolist()
