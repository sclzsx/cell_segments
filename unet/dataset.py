import os
import cv2
from PIL import Image
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import enet_weighing, median_freq_balancing
import torch.nn as nn
from collections import OrderedDict, Counter
import random
from utils import add_mask_to_source_multi_classes, add_mask_to_source
from pathlib import Path


def get_class_weights(loader, out_channels, weighting):
    print('Weighting method is:{}, please wait.'.format(weighting))
    if weighting == 'enet':
        class_weights = enet_weighing(loader, out_channels)
        class_weights = torch.from_numpy(class_weights).float().cuda()
    elif weighting == 'mfb':
        class_weights = median_freq_balancing(loader, out_channels)
        class_weights = torch.from_numpy(class_weights).float().cuda()
    else:
        class_weights = None
    return class_weights


class PILToLongTensor(object):
    def __call__(self, pic):
        if not isinstance(pic, Image.Image):
            raise TypeError("pic should be PIL Image. Got {}".format(
                type(pic)))
        # handle numpy array
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.long()
        # Convert PIL image to ByteTensor
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # Reshape tensor
        nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # Convert to long and squeeze the channels
        return img.transpose(0, 1).transpose(0, 2).contiguous().long().squeeze_()


class SegDataset(Dataset):
    def __init__(self, dataset_dir, num_classes=2, appoint_size=(512, 512), erode=0, aug=False):
        self.label_paths = [str(i) for i in Path(dataset_dir).rglob('*_mask.png')]
        self.num_classes = num_classes
        self.appoint_size = appoint_size
        self.erode = erode
        self.aug = aug

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, i):
        label_path = self.label_paths[i]
        img_path = label_path.replace('_mask', '')

        image = Image.open(img_path).convert('L')
        label = Image.open(label_path)

        image = image.resize(self.appoint_size)
        label = label.resize(self.appoint_size, Image.NEAREST)

        if self.aug:
            if np.random.rand() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.rand() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                label = label.transpose(Image.FLIP_TOP_BOTTOM)

        label_np = np.array(label)
        label_np = np.where(label_np > 0, 1, 0)
        label = Image.fromarray(label_np)

        transform = transforms.ToTensor()

        img_tensor = transform(image)
        label_tensor = transform(label).long()

        check = False
        if check:
            label_check = np.array(label_tensor)
            label_dict = Counter(label_check.flatten())
            label_list = [j for j in range(self.num_classes)]
            for k, v in label_dict.items():
                if k not in label_list:
                    print(img_path, label_path, label_dict)
            print(label_dict)
            print(img_tensor.shape, label_tensor.shape, img_tensor.dtype, label_tensor.dtype)
            print(torch.min(img_tensor), torch.max(img_tensor), torch.min(label_tensor), torch.max(label_tensor))
        return img_tensor, label_tensor


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    aug = True
    num_classes = 2
    appoint_size = (256, 256)
    dataset_dir = '../../datasets/model1_training'

    dataset = SegDataset(dataset_dir, num_classes=num_classes, appoint_size=appoint_size, erode=0, aug=aug)

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    for i, batch_data in enumerate(loader):
        if i % 100 == 0:
            print('Check done', i)
