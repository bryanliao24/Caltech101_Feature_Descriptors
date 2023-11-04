import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from torchvision.io import read_image, ImageReadMode
import torchvision 
from torchvision import datasets, models, transforms 
from pathlib import Path 
from PIL import Image
import os 
import cv2
import numpy as np
from PIL import Image

def compute_color_moments(cell):
    # Calculate color moments for each channel (R, G, B)
    moments = []
    
    for channel in range(3):  # Loop over R, G, B channels
        # [:, :, channel]: This is the slicing syntax in NumPy.
        # for the first include all row
        # for the seconda include all col
       
        channel_data = cell[:, :, channel]
        # when encounter data = NaN or 0 which may cause divided error -> Two ways to avoid
        
        # 1. Check for Zero Variance -> esure that there is sufficient variance in the data
        mean = np.mean(channel_data)
        std_dev = np.std(channel_data)
        if std_dev < 1e-6:  # an appropriate threshold
            skewness = 0 
        else:
            skewness = np.mean(((channel_data - mean) / std_dev) ** 3)
            
#         # 2. Handling NaN Values -> to use np.isnan to identify and filter out NaN values:
#         valid_data = channel_data[~np.isnan(channel_data)]
#         mean = np.mean(valid_data)
#         std_dev = np.std(valid_data)
#         skewness = np.mean(((valid_data - mean) / std_dev) ** 3)

        #kewness = np.mean(((channel_data - mean) / std_dev) ** 3)  # Skewness formula

        # Append the calculated moments for the channel
        moments.extend([mean, std_dev, skewness])

    return moments



def cal_colorMoment(img):
    # Assuming 'image' is your PIL image
    image_np = np.array(img)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    grid_size = (10, 10)
    image = cv2.resize(image_cv2, (300, 100))
    height, width, channel = image.shape
    cell_height = height // grid_size[0]
    cell_width = width // grid_size[1]

    # store color moments for each grid cell
    color_moments = []

    # use slice to interpreted the index element
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Extract the current cell from the image
            cell = image[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]
            cell_color_moments = compute_color_moments(cell)
            color_moments.append(cell_color_moments)
     # Convert the 'color_moments' list to a NumPy array
    feature_descriptor_CM = np.array(color_moments).reshape(grid_size[0], grid_size[1], 3, 3)
    feature_descriptor_CM = feature_descriptor_CM.reshape(-1)  # Flatten to a 1D array (900-dimensional feature descriptor)

    #print("Feature Descriptor_CM:")
    #print(feature_descriptor_CM)
    return feature_descriptor_CM
