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
import pickle
from Color_moment import cal_colorMoment
from HOG import cal_HOG
from Avgpool import cal_ResNet_AvgPool
from Layer3 import cal_ResNet_Layer3
from FC import cal_ResNet_FC
# Assuming you have dictionaries for each model's data
color_moments_data = {}
hog_data = {}
resnet_avgpool_data = {}
resnet_layer3_data = {}
resnet_fc_data = {}

dataset = torchvision.datasets.Caltech101(r'C:\Users\USER\Desktop\Caltech101', download=False)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=8) 

# Loop to fill the data dictionaries for each model (replace with your actual data retrieval)
for image_ID in range(8677):
    img, Label = dataset[image_ID] 
    image_np = np.array(img)
    color_moments_data[image_ID] = cal_colorMoment(img)
    hog_data[image_ID] = cal_HOG(img)
    resnet_avgpool_data[image_ID] = cal_ResNet_AvgPool(image_np)
    resnet_layer3_data[image_ID] = cal_ResNet_Layer3(image_np)
    resnet_fc_data[image_ID] = cal_ResNet_FC(image_np)


# Specify the pickle file names for each model
color_moments_pickle_file = 'color_moments_data.pickle'
hog_pickle_file = 'hog_data.pickle'
resnet_avgpool_pickle_file = 'resnet_avgpool_data.pickle'
resnet_layer3_pickle_file = 'resnet_layer3_data.pickle'
resnet_fc_pickle_file = 'resnet_fc_data.pickle'

# Function to save data to pickle file
def save_data_to_pickle(data, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)

# Save each model's data to its respective pickle file
save_data_to_pickle(color_moments_data, color_moments_pickle_file)
save_data_to_pickle(hog_data, hog_pickle_file)
save_data_to_pickle(resnet_avgpool_data, resnet_avgpool_pickle_file)
save_data_to_pickle(resnet_layer3_data, resnet_layer3_pickle_file)
save_data_to_pickle(resnet_fc_data, resnet_fc_pickle_file)

print("Data saved to pickle files.")