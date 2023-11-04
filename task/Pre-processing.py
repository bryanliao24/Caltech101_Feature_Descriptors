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
# from IPython.display import display, Image 
print("Torchvision Version: ",torchvision.__version__)

import torch
dataset = torchvision.datasets.Caltech101(r'C:\Users\USER\Desktop\Caltech101', download=False)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=8) 
print(len(dataset))

for image_ID in range(8677):    
    img, label = dataset[image_ID]    
    print(image_ID)    
    newsize = (60, 60)    
    img = img.resize(newsize)    

# # type the given image_id he

# print("image_id:", image_id)
# newsize = (200, 150)    
# imgg = imgg.resize(newsize)    
# i = np.array(imgg)
# print(i.shape)
