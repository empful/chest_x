# from torchvision import models

# model = models.densenet121()
# print(model.features)

from cxr_dataset import CXRDataset
import torch
from tqdm import tqdm
import numpy as np
import random

# # placeholders
# psum    = torch.tensor([0.0, 0.0, 0.0])
# psum_sq = torch.tensor([0.0, 0.0, 0.0])

# PATH_TO_IMAGES = "/mnt/bd/medai-cv/cvpr/CXR8/images/images"
# dataset = CXRDataset(path_to_images=PATH_TO_IMAGES,
#         fold='train')

# random.seed(1)
# randomlist = random.sample(range(len(dataset)), 10000)

# means = []
# for sample in tqdm(randomlist):
#     img, label, _ = dataset.__getitem__(sample)
#     img = np.array(img.convert('L'))/255.
#     means.append(np.mean(img))
# mu = np.mean(means)

# variances = []
# for sample in tqdm(randomlist):
#     img, label, _ = dataset.__getitem__(sample)
#     img = np.array(img.convert('L'))/255.
#     var = np.mean((img - mu) ** 2)
#     variances.append(var)
# std = np.sqrt(np.mean(variances)) 

# print(mu, std)



import json

with open("debug_original.json","r") as f:
    data= json.load(f)

losses = [value for key,value in data.items()]
l=len(losses)
print(np.mean(losses))
print(losses[l-50:])
print(l)