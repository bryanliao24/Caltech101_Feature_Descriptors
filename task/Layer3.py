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

def cal_ResNet_Layer3(image_np):
    if image_np.shape[-1] != 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = preprocess(image_np).unsqueeze(0)
    model = models.resnet50(pretrained=True)
    model.eval()
    def hook(module, input, output):
        hook_output.append(output)

    hook_output = []
    model.layer3.register_forward_hook(hook)
    with torch.no_grad():
        model.eval()
        output = model(image)

    output_tensor = hook_output[0]

    # Average each 14x14 slice along the dimensions to obtain a 1024-dimensional vector
    feature_vector = torch.mean(output_tensor, dim=(2, 3))

    #print("Feature Descriptor (ResNet-Layer3-1024):")
    #print(feature_vector)
    return feature_vector