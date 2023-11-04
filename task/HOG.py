from torchvision.io import read_image, ImageReadMode
import torchvision 
from torchvision import datasets, models, transforms 
from pathlib import Path 
from PIL import Image
import os 
import cv2
import numpy as np
from PIL import Image
def cal_HOG(img):
    # Load your image
    image_np = np.array(img)
    if image_np.shape[-1] != 3:
        color_image = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    # Convert the image to grayscale
    else:
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    resized_image = cv2.resize(gray_image, (300, 100))

    # Compute gradients using the (-1,0,1) and (-1,0,1) transpose masks
    dx_mask = np.array([[-1, 0, 1]])
    dy_mask = np.array([[-1], [0], [1]])
    gradient_x = cv2.filter2D(resized_image.astype(np.float64), -1, dx_mask).astype(np.float64)
    gradient_y = cv2.filter2D(resized_image.astype(np.float64), -1, dy_mask).astype(np.float64)

    # Calculate gradient magnitude and angle
    magnitude, angle = cv2.cartToPolar(gradient_x, gradient_y, angleInDegrees=True)

    # Ensure angles are in the range [0, 360)
    angle[angle < 0] += 360 

    # Define the number of cells in the x and y directions
    num_cells_x = 10
    num_cells_y = 10

    # Define the number of bins for the histogram
    num_bins = 9

    # Initialize the HOG feature vector
    feature_descriptor_HOG = np.zeros((num_cells_x, num_cells_y, num_bins), dtype=np.float32)

    # Define custom bin boundaries for HOG histogram bins
    bin_boundaries = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360]

    cell_size_x = magnitude.shape[1] // num_cells_x
    cell_size_y = magnitude.shape[0] // num_cells_y

    for i in range(num_cells_x):
        for j in range(num_cells_y):
            cell_magnitude = magnitude[j * cell_size_y:(j + 1) * cell_size_y, i * cell_size_x:(i + 1) * cell_size_x]
            cell_angle = angle[j * cell_size_y:(j + 1) * cell_size_y, i * cell_size_x:(i + 1) * cell_size_x]

            # Calculate the histogram for the cell
            hist, _ = np.histogram(cell_angle, bins=bin_boundaries, range=(0, 360), weights=cell_magnitude)

            # Store the histogram in the HOG descriptor
            feature_descriptor_HOG[i, j, :] = hist

    # Flatten the HOG descriptor to a 900-dimensional vector
    feature_descriptor_HOG = feature_descriptor_HOG.flatten()
    #print(feature_descriptor_HOG)
    return feature_descriptor_HOG
    # Print the shape of the HOG descriptor
