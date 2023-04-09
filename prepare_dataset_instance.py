import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import labelme
import numpy as np
import cv2
import random
import os
from tqdm import tqdm
import PIL
from collections import Counter


def labelme_json_to_mask_binary(data_dir):
    json_paths = [i for i in Path(data_dir).rglob('*.json')]

    for json_path in json_paths:
        json_path = str(json_path)
        image_path = json_path.replace('.json', '.png')
        mask_path = image_path[:-4] + '_mask.png'

        image = Image.open(image_path)
        imageHeight = image.height
        imageWidth = image.width
        img_shape = (imageHeight, imageWidth)

        with open(json_path, 'r', encoding='gb18030', errors='ignore') as f:
            data = json.load(f)

        try:
            mask, _ = labelme.utils.shape.labelme_shapes_to_label(img_shape, data['shapes'])
            mask = np.array(mask).astype('uint8')
            mask = np.where(mask > 0, 255, 0).astype('uint8')
            cv2.imwrite(mask_path, mask)
        except:
            print('Error:', json_path)
            pass


def json_to_instance(json_path):
    json_path = str(json_path)
    image_path = json_path.replace('.json', '.png')
    mask_path = image_path[:-4] + '_mask.png'
    if os.path.exists(mask_path):
        print('Pass as Existed', mask_path)
        return

    image = Image.open(image_path)
    imageHeight = image.height
    imageWidth = image.width
    img_shape = (imageHeight, imageWidth)

    with open(json_path, 'r', encoding='gb18030', errors='ignore') as f:
        data = json.load(f)

    instance_mask = np.zeros(img_shape, dtype=np.uint16)
    shapes = data['shapes']
    ins_id = 1
    for shape in shapes:
        if shape['label'] == 'cell':
            try:
                points = shape['points']
                mask = np.zeros(img_shape, dtype=np.uint8)
                mask = PIL.Image.fromarray(mask)
                draw = PIL.ImageDraw.Draw(mask)
                xy = [tuple(point) for point in points]
                draw.polygon(xy=xy, outline=1, fill=1)
                mask = np.array(mask, dtype=bool)
                instance_mask[mask] = ins_id
                ins_id += 1
            except:
                print('########## Pass as Error in', json_path)
    instance_mask = PIL.Image.fromarray(instance_mask)
    instance_mask.save(mask_path)

    check = False
    if check:
        tmp = np.array(PIL.Image.open(mask_path))
        print(tmp.shape, tmp.dtype)
        cnt = Counter(tmp.flatten())
        print(cnt)
        K = []
        for k, v in cnt.items():
            K.append(k)
        K.sort()
        print(K)
        tmp[tmp > 300] = 0
        plt.imshow(tmp)
        plt.show()


def labelme_json_to_mask(data_dir):
    json_paths = [i for i in Path(data_dir).rglob('*.json')]

    for json_path in tqdm(json_paths):
        json_to_instance(json_path)


def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def split_train_test(data_dir, test_rate, dst_size):
    mask_paths = [i for i in Path(data_dir).rglob('*_mask.png')]
    random.shuffle(mask_paths)
    test_num = int(len(mask_paths) * test_rate)

    save_dir_train_images = data_dir + '_split/train/images'
    save_dir_test_images = data_dir + '_split/test/images'
    save_dir_train_masks = data_dir + '_split/train/masks'
    save_dir_test_masks = data_dir + '_split/test/masks'
    mkdir(save_dir_train_images)
    mkdir(save_dir_test_images)
    mkdir(save_dir_train_masks)
    mkdir(save_dir_test_masks)

    cnt = 0
    for mask_path in tqdm(mask_paths):
        name = mask_path.name.replace('_mask', '')
        mask_path = str(mask_path)
        image_path = mask_path.replace('_mask', '')
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        if dst_size is not None:
            image = cv2.resize(image, dst_size)
            mask = cv2.resize(mask, dst_size, interpolation=cv2.INTER_NEAREST)

        if cnt < test_num:
            cv2.imwrite(save_dir_test_images + '/' + name, image)
            cv2.imwrite(save_dir_test_masks + '/' + name, mask)
        else:
            cv2.imwrite(save_dir_train_images + '/' + name, image)
            cv2.imwrite(save_dir_train_masks + '/' + name, mask)

        cnt += 1


if __name__ == '__main__':
    data_dirs = [
        '../datasets/model1_training',
        '../datasets/model2_training',
        '../datasets/model3_training',

        '../datasets/model1_test',
        '../datasets/model2_test',
        '../datasets/model3_test',
    ]

    for data_dir in data_dirs:
        labelme_json_to_mask(data_dir)
