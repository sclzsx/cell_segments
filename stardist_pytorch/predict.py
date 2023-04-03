# SPDX-FileCopyrightText: 2022 Ashis Ravindran <ashis(dot)r91(at)gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from glob import glob
from tifffile import imread
from csbdeep.utils import normalize
from stardist import dist_to_coord, non_maximum_suppression, polygons_to_label
from stardist import random_label_cmap,ray_angles
import metric
import torch
from unet import UNetStar as UNet
import os
from skimage import io
from pathlib import Path

# DATASET_PATH_IMAGE = 'datasets/dsb2018_kaggle_split/test/images/*.png'
# DATASET_PATH_LABEL = 'datasets/dsb2018_kaggle_split/test/masks/*.png'
MODEL_WEIGHTS_PATH= 'CHECKPOINT/checkpoint_UNet2D_StarDist_DSB2018/CHECKPOINT.t7'


# X = sorted(glob(DATASET_PATH_IMAGE))
# X = list(map(imread,X))
# Y = sorted(glob(DATASET_PATH_LABEL))
# Y = list(map(imread,Y))

# n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
# axis_norm = (0,1)   # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
# if n_channel > 1:
#     print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
N_RAYS = 32
angles = ray_angles(N_RAYS)

def plot(target_label,pred_label):
    plt.figure(figsize=(16,10))
    plt.subplot(211)
    plt.imshow(target_label.squeeze(),cmap=random_label_cmap())
    plt.axis('off')
    plt.title('Ground truth')
    plt.subplot(212)
    plt.axis('off')
    plt.imshow(pred_label,cmap=random_label_cmap())
    plt.title('Predicted Label.')
    plt.show()
        
def predictions(model_dist, X):
    img = normalize(X,1,99.8,axis=(0, 1))
    input = torch.tensor(img)
    input = input.unsqueeze(0).unsqueeze(0)#unsqueeze 2 times
    dist,prob = model_dist(input)
    dist_numpy= dist.detach().cpu().numpy().squeeze()
    prob_numpy= prob.detach().cpu().numpy().squeeze()
    return dist_numpy,prob_numpy
    
model_dist = UNet(1,N_RAYS)
model_dist.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
print('Distance weights loaded')

apscore_nms = []
prob_thres = 0.4

images_dir = 'datasets/dsb2018_kaggle_split/test/images'

for idx,img_path in enumerate(Path(images_dir).glob('*.png')):
    print(idx)
    
    img_path = str(img_path)
    mask_path = img_path.replace('images', 'masks')

    image = io.imread(img_path)
    target = io.imread(mask_path)
    target = np.where(target > 0, 1, 0)
    print(image.shape, target.shape)

    dists,probs=predictions(model_dist,image)
    dists = np.transpose(dists,(1,2,0))
    coord = dist_to_coord(dists)
    points = non_maximum_suppression(coord,probs,prob_thresh=prob_thres)
    star_label = polygons_to_label(coord,probs,points)
    apscore_nms.append(metric.calculateAPScore(star_label,target,IOU_tau=0.5))
    #plot(target,star_label) 
print('Total images',idx+1)
ap_nms = sum(apscore_nms)/(len(apscore_nms))
print('AP NMS',ap_nms)
   