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

def cal_ResNet_FC(image_np):
    if image_np.shape[-1] != 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    resnet = models.resnet50(pretrained=True)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = preprocess(image_np).unsqueeze(0) 

    def hook_fn(module, input, output):
        global fc_output
        fc_output = output

    # Attach the hook to the fully connected layer
    hook = resnet.fc.register_forward_hook(hook_fn)

    with torch.no_grad():
        resnet(image)
    hook.remove()
    
    #print("Feature Descriptor (ResNet-FC-1000):")
    #print(fc_output)
    return fc_output