import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil

def compute_gradient_magnitude_color(image):
    """Compute gradient magnitude for each color channel and combine the results."""
    channels = cv2.split(image)
    
    grad_magnitudes = []
    
    # Compute gradient magnitude for each channel
    for channel in channels:
        grad_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_magnitudes.append(grad_magnitude)
    
    # Combine the gradient magnitudes by taking the maximum gradient at each pixel
    combined_grad_magnitude = np.max(np.array(grad_magnitudes), axis=0)
    return np.mean(combined_grad_magnitude)

def get_images_with_resolution(folder_path, resolution=(500, 500), max_images=100):
    """Get images that match the specified resolution and copy them to the ./ folder."""
    valid_extensions = ('.JPEG')
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(valid_extensions)]
    
    selected_images = []
    gradients = []
    
    # Go through images and select those with the required resolution
    for image_path in image_paths:
        if len(selected_images) >= max_images:
            break
        
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        # Print the image name and its resolution
        print(f"Checking {os.path.basename(image_path)}: Resolution = {width}x{height}")
        
        # Check if the image matches the desired resolution
        if height == resolution[0] and width == resolution[1]:
            grad_magnitude = compute_gradient_magnitude_color(image)
            selected_images.append((image_path, grad_magnitude))
            gradients.append(grad_magnitude)
            
    return selected_images, gradients

def move_image_to_fidelity_folder(image_path, grad_magnitude, low_threshold, high_threshold):
    """Move an image to the appropriate folder based on its gradient magnitude."""
    image_name = os.path.basename(image_path)

    # Determine the target folder based on thresholds
    if grad_magnitude < low_threshold:
        target_folder = './low'
    elif low_threshold <= grad_magnitude < high_threshold:
        target_folder = './medium'
    else:
        target_folder = './high'

    # Create the folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Move the image to the target folder
    shutil.move(image_path, os.path.join(target_folder, image_name))
    print(f"Moved {image_name} to {target_folder}")

def plot_gradient_distribution(gradients):
    """Plot the distribution of gradient magnitudes."""
    plt.hist(gradients, bins=20, color='blue', edgecolor='black')
    plt.title('Distribution of Gradient Magnitudes (500x500 Images)')
    plt.xlabel('Gradient Magnitude')
    plt.ylabel('Number of Images')
    plt.show()

if __name__ == "main":
    """
    This script can be used to seperate your dataset of images based on the fidelity of the images.
    """

    # Download a dataset with images and point to it
    folder_path = 'C:\\Users\\user\Downloads\\images'

    # Get 100 images of 500x500 resolution
    selected_images, gradients = get_images_with_resolution(folder_path, resolution=(500, 500), max_images=300)

    # Plot the distribution of gradients to manually set thresholds
    if gradients:
        plot_gradient_distribution(gradients)
    else:
        print("No images of the specified resolution were found.")

    # Manually define thresholds for low, medium, and high fidelity after viewing the plot
    low_threshold = 70    # Example threshold, modify based on the distribution
    high_threshold = 200   # Example threshold, modify based on the distribution

    # Move images to the appropriate folders
    for image_path, grad_magnitude in selected_images:
        move_image_to_fidelity_folder(image_path, grad_magnitude, low_threshold, high_threshold)


