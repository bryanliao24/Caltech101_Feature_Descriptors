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

def cal_ResNet_AvgPool(image_np):
    if image_np.shape[-1] != 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    
    # image = preprocess(Image.open(image_path)).unsqueeze(0)  # Add batch dimension
    image = preprocess(image_np).unsqueeze(0)
    # Load the pre-trained ResNet50 model with default weights
    model = models.resnet50(pretrained=True)
    model.eval()

    # Define a hook to capture the output of the avgpool layer
    def hook(module, input, output):
        hook_output.append(output)

    # the hook have to register and then capture the averagepool layer
    hook_output = []
    model.avgpool.register_forward_hook(hook)

    # Forward pass
    with torch.no_grad():
        model.eval()
        output = model(image)

    # Extract the feature vector
    feature_vector = hook_output[0].squeeze()

    # Reduce the dimensionality to 1024 by averaging two consecutive entries mentioned in the PDF
    reduced_feature_vector = torch.mean(feature_vector.view(-1, 2, 1024), dim=1)

    #print("Feature Descriptor (ResNet-AvgPool-1024):")
    #print(reduced_feature_vector)
    return reduced_feature_vector