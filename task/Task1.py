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



import torch
dataset = torchvision.datasets.Caltech101(r'C:\Users\USER\Desktop\Caltech101', download=False)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=8) 


for image_ID in range(8677):    
    img, label = dataset[image_ID]    
   

print("Please input the Image_ID you want to compute:")
idd = input()
iddd = int(idd)
imgg, label = dataset[iddd]  
image_np = np.array(imgg)

CM_DB = cal_colorMoment(image_np)
print("color moment is :", CM_DB)
HOG_DB = cal_HOG(image_np)
print("HOG is :", HOG_DB)
AVG_DB = cal_ResNet_AvgPool(image_np)
print("avgpool is :", AVG_DB)
LAY3_DB = cal_ResNet_Layer3(image_np)
print("layer3 is :", LAY3_DB)
FC_DB = cal_ResNet_FC(image_np)
print("FC-layer is :", FC_DB)

