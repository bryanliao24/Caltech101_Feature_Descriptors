import pickle
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
dataset = torchvision.datasets.Caltech101(r'C:\Users\USER\Desktop\Caltech101', download=False)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=8) 

# Specify the path to your pickle file
color_moments_pickle_file = 'color_moments_data.pickle'
hog_pickle_file = 'hog_data.pickle'
resnet_avgpool_pickle_file = 'resnet_avgpool_data.pickle'
resnet_layer3_pickle_file = 'resnet_layer3_data.pickle'
resnet_fc_pickle_file = 'resnet_fc_data.pickle'

# Open the pickle file for reading in binary mode
with open(color_moments_pickle_file, 'rb') as file:
    # Load the data from the pickle file
    color_moment_data = pickle.load(file)
    # HOG_data = pickle.load(file)
with open(hog_pickle_file, 'rb') as file1:
    # Load the data from the pickle file
    # color_moment_data = pickle.load(file)
    HOG_data = pickle.load(file1)
with open(resnet_avgpool_pickle_file, 'rb') as file2:
    avgpool_data = pickle.load(file2)
with open(resnet_layer3_pickle_file, 'rb') as file3:
    layer3_data = pickle.load(file3)
with open(resnet_fc_pickle_file, 'rb') as file4:
    fc_data = pickle.load(file4)

# print(color_moment_data[0])
#print(HOG_data[8676])
# # print(len(HOG_data))
# input the target image_id
print(fc_data[8676])


import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Calculate color moment feature descriptor for the first image (image_id 0)
target_image_id = 5122
target_color_moment = color_moment_data[target_image_id]

# Calculate color moment feature descriptors for images 1 to 20
image_ids_to_compare = range(1, 8677)
color_moments_to_compare = [color_moment_data[i] for i in image_ids_to_compare]


def euclidean_distance(descriptor1, descriptor2):
    return np.linalg.norm(descriptor1 - descriptor2)

# Define a function to compute KL divergence
# Define a small epsilon value to avoid division by zero
epsilon = 1e-10
def kl_divergence(descriptor1, descriptor2):
    descriptor1 = (descriptor1 + epsilon) / np.sum(descriptor1 + epsilon)
    descriptor2 = (descriptor2 + epsilon) / np.sum(descriptor2 + epsilon)
    return np.sum(descriptor1 * np.log(descriptor1 / descriptor2))

## distances = [cosine_similarity(np.array(target_color_moment).reshape(1, -1), np.array(cm).reshape(1, -1)) for cm in color_moments_to_compare]
def cosin_similarity(descriptor1, descriptor2):
    # Calculate the dot product of the two vectors
    dot_product = np.dot(descriptor1, descriptor2)
    
    # Calculate the magnitudes of each vector
    magnitude1 = np.linalg.norm(descriptor1)
    magnitude2 = np.linalg.norm(descriptor2)
    
    # Calculate the cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)
    
    return similarity

def quadratic_distance(descriptor1, descriptor2):
    return np.sum((descriptor1 - descriptor2) ** 2)

from scipy.stats import wasserstein_distance
# earth move distance (EMD)

from scipy.spatial.distance import minkowski

def intersection_similarity(descriptor1, descriptor2):
    # Convert descriptors to sets
    set1 = set(descriptor1)
    set2 = set(descriptor2)
    
    # Calculate the size of intersection and union
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))
    
    # Calculate the Intersection similarity
    similarity = intersection_size / union_size
    
    return similarity

def pearson_correlation(descriptor1, descriptor2):
    # Calculate the means of both vectors
    mean1 = np.mean(descriptor1)
    mean2 = np.mean(descriptor2)
    
    # Calculate the numerator and denominators
    numerator = np.sum((descriptor1 - mean1) * (descriptor2 - mean2))
    denominator1 = np.sqrt(np.sum((descriptor1 - mean1) ** 2))
    denominator2 = np.sqrt(np.sum((descriptor2 - mean2) ** 2))
    
    # Calculate the Pearson's correlation coefficient
    correlation_coefficient = numerator / (denominator1 * denominator2)
    
    return correlation_coefficient

# Compute Euclidean distances between the target image and images 1 to 20
distances = [euclidean_distance(target_color_moment, cm) for cm in color_moments_to_compare]

#distances = [cosine_similarity(np.array(target_color_moment).reshape(1, -1), np.array(cm).reshape(1, -1)) for cm in color_moments_to_compare]


# Create a list of (image_id, distance) pairs
image_distances = list(zip(image_ids_to_compare, distances))

# Sort the pairs by distance (ascending order)
sorted_image_distances = sorted(image_distances, key=lambda x: x[1])

# Select the top 3 most similar images
top_10_similar_images = sorted_image_distances[:13]
#top_10_similar_images = sorted_image_distances[-10:]

imgg1, Label = dataset[target_image_id] 
newsize = (200, 150)    
imgg1 = imgg1.resize(newsize) 

# Print the top 3 similar images and their distances
for image_id, distance in top_10_similar_images:
    print(f"Image ID: {image_id}, Distance: {distance/10}")
    newsize = (200, 150)
    img, Label = dataset[image_id] 
    img1 = img.resize(newsize)    

def par():
    # resnet_fc_data
    # resnet_layer3_data
    with open('resnet_fc_data.pickle', 'rb') as file:
        resnet_avgpool_data = pickle.load(file)

    # Define the target image_id
    target_image_id = 8676 # Change this to the desired image_id

    # Get the target feature descriptor
    target_descriptor = np.array(resnet_avgpool_data[target_image_id])

    # Initialize a dictionary to store correlations
    correlations = {}

    # Calculate Pearson's correlation coefficient for the target image with all other images
    for image_id, descriptor in resnet_avgpool_data.items():
        if image_id != target_image_id:
            correlation = np.corrcoef(target_descriptor, descriptor)[0, 1]
            correlations[image_id] = correlation

    # Sort the correlations to find the 10 most similar images
    sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    top_10_similar_images = sorted_correlations[:12]

    imgg1, Label = dataset[target_image_id] 
    newsize = (200, 150)    
    imgg1 = imgg1.resize(newsize) 
    
    # Print the top 3 similar images and their distances
    for image_id, correlation in top_10_similar_images:
        print(f"Image ID: {image_id}, Distance: {correlation}")
        newsize = (200, 150)
        img, Label = dataset[image_id] 
        img1 = img.resize(newsize)    
        
   