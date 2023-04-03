from pathlib import Path
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from collections import Counter
import random
from tqdm import tqdm
import os

def check_dsb2018_kaggle(data_root='datasets/dsb2018_kaggle'):
    for image_path in Path(data_root + '/images_all').glob('*.png'):
        image_path = str(image_path)
        label_path = image_path.replace('images', 'labels').replace('png', 'tif')
        mask_path = image_path.replace('images', 'masks')
        image = Image.open(image_path)
        label = Image.open(label_path)
        mask = Image.open(mask_path)
        image = np.array(image)
        label = np.array(label)
        mask = np.array(mask)

        label_count = Counter(label.flatten())
        mask_count = Counter(mask.flatten())

        # print(image.shape, label.shape, mask.shape)
        # print('label_count', label_count)
        # print('mask_count', mask_count)

        assert (len(mask_count) == 2)
        # plt.imshow(image)
        # plt.show()

        labels = [k for k,v in label_count.items()]
        labels.sort()
        print(labels)

def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def split_dsb2018_kaggle(data_root='datasets/dsb2018_kaggle', test_rate=0.3):
    image_paths = [image_path for image_path in Path(data_root + '/images_all').glob('*.png')]
    random.shuffle(image_paths)
    test_num = int(len(image_paths) * test_rate)
    cnt = 0
    for image_path in tqdm(image_paths):
        image_path = str(image_path)
        label_path = image_path.replace('images', 'labels').replace('png', 'tif')
        mask_path = image_path.replace('images', 'masks')
        image = Image.open(image_path)
        label = Image.open(label_path)
        mask = Image.open(mask_path)
        image = np.array(image)
        label = np.array(label)
        mask = np.array(mask)
        h, w, c = image.shape
        gray = np.zeros([h, w])
        for i in range(c):
            gray = gray + image[:,:,i]
        gray = (gray / c).astype('uint8')

        mask = mask[:,:,2] # blue
        
        gray = Image.fromarray(gray)
        mask = Image.fromarray(mask)
        gray = gray.resize((256, 256))
        mask = mask.resize((256, 256), Image.NEAREST)

        if cnt < test_num:
            image_dir = data_root + '_split/test/images'
            mask_dir = data_root + '_split/test/masks'
            mkdir(image_dir)
            mkdir(mask_dir)
            gray.save(image_dir + '/' + str(cnt) + '.png')
            mask.save(mask_dir + '/' + str(cnt) + '.png')
        else:
            image_dir = data_root + '_split/train/images'
            mask_dir = data_root + '_split/train/masks'
            mkdir(image_dir)
            mkdir(mask_dir)
            gray.save(image_dir + '/' + str(cnt) + '.png')
            mask.save(mask_dir + '/' + str(cnt) + '.png')

        cnt += 1

def check_dsb2018_kaggle_split(data_root='datasets/dsb2018_kaggle'):
    for image_path in Path(data_root + '_split').rglob('*.png'):
        image_path = str(image_path)
        if 'images' not in image_path:
            continue

        mask_path = image_path.replace('images', 'masks')
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        image = np.array(image)
        mask = np.array(mask)
        print(image.shape, mask.shape)

        mask_count = Counter(mask.flatten())
        assert (len(mask_count) == 2)


if __name__ == '__main__':
    # check_dsb2018_kaggle()
    # split_dsb2018_kaggle()
    check_dsb2018_kaggle_split()