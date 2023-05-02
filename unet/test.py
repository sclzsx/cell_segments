from pathlib import Path
from argparse import ArgumentParser
import torch
import json
from choices import choose_net
from predictor import eval_dataset_full, predict_images
from dataset import SegDataset, get_class_weights
from torch.utils.data import DataLoader
import os
import cv2
import numpy as np
from torchvision import transforms
from matplotlib import pyplot as plt
from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available, _draw_polygons
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D
from tqdm import tqdm
import sys
from PIL import Image
import time


def get_test_args():
    parser = ArgumentParser()
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--out-channels", type=int)
    parser.add_argument("--erode", type=int, default=0)
    parser.add_argument("--pt-dir", type=str)
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--test-videos", type=str)
    parser.add_argument("--test-set", type=str)
    parser.add_argument("--test-images", type=str)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--weighting", type=str, default='none')
    parser.add_argument("--pt-root", type=str, default='./Results/')
    parser.add_argument("--vis", type=bool, default=True)
    return parser.parse_args()


def find_latest_pt(dir):
    max_num = 0
    latest_path = ''
    for path in Path(dir).glob('*.pt'):
        # print(path)
        num = int(path.name.split('_')[-1].split('.')[0])
        if num > max_num:
            max_num = num
            latest_path = str(path)
    if latest_path != '':
        return latest_path
    else:
        print('No pts this dir:', dir)
        return None


def merge_args_from_train_json(args, json_path, verbose=False):
    if not os.path.exists(json_path):
        return args
    with open(json_path, 'r') as f:
        train_d = json.load(f)
        if verbose:
            print(train_d)
    args.weighting = train_d['weighting']
    args.dilate = train_d['erode']
    args.net_name = train_d['net_name']
    args.out_channels = train_d['out_channels']
    args.save_suffix = train_d['save_suffix']
    args.height = train_d['height']
    args.width = train_d['width']
    with open(json_path.replace('train_args', 'test_args'), 'w') as f:
        d = vars(args)
        json.dump(d, f, indent=2)
    if verbose:
        for k, v in d.items():
            print(k, v)
    return args


def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def segment_binary_to_instance(output):
    binary, contours, hir = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    h, w = output.shape
    output_ins = np.zeros((h, w), dtype='int32')

    # output_ins_vis = np.zeros((h, w, 3), dtype='uint8')
    cnt = 1

    areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        areas.append(area)
    ids = np.argsort(areas)[::-1]

    if len(areas) > 0:
        mid_id = ids[len(areas) // 2]
        mid_area = areas[mid_id]
        # print('mid_area:', mid_area)

        for id in ids:
            contour = contours[id]

            area = cv2.contourArea(contour)
            if area < mid_area * 0.2 or area > mid_area * 1.8:
                continue

            mask = np.zeros_like(output)
            ret = cv2.drawContours(mask, [contour], -1, 255, -1)
            output_ins[mask > 0] = cnt

            cnt += 1
        # print(cnt)
    return output_ins


def do_test(args):
    mkdir(args.save_dir)

    pt_dir = args.pt_root + '/' + args.pt_dir
    args = merge_args_from_train_json(args, json_path=pt_dir + '/train_args.json')
    # pt_path = find_latest_pt(pt_dir)
    pt_path = pt_dir + '/unet_best.pt'
    print('Loading:', pt_path)
    net = choose_net(args.net_name, args.out_channels).cuda()
    net.load_state_dict(torch.load(pt_path))
    net.eval()

    image_size = 256

    Y, Y_pred, FPS = [], [], []
    paths = [i for i in Path(args.test_images).glob('*_mask.png')]
    cnt = 0
    for label_path in tqdm(paths):
        label_path = str(label_path)
        img_path = label_path.replace('_mask', '')

        image_color = Image.open(img_path)
        image = image_color.convert('L')

        label = Image.open(label_path)

        image = image.resize((image_size, image_size))
        label = label.resize((image_size, image_size), Image.NEAREST)
        label_np = np.array(label)

        transform = transforms.ToTensor()
        img_tensor = transform(image)

        time0 = time.time()
        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(0).cuda()
            out_tensor = net(img_tensor)
            output = np.array(torch.max(out_tensor.data, 1)[1].squeeze().cpu()).astype('uint8')
        output_ins = segment_binary_to_instance(output)
        # plt.imshow(output_ins)
        # plt.show()
        time1 = time.time()

        image_color = image_color.resize((image_size, image_size))
        image_color = np.array(image_color)
        image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)

        pred_mask = output_ins.astype('bool')
        pred_mask = np.expand_dims(pred_mask, axis=-1)
        pred_color = image_color * pred_mask
        save_dir = args.save_dir + '/predsColor' + str(image_size)
        mkdir(save_dir)
        cv2.imwrite(save_dir + '/' + str(cnt) + '.png', pred_color)

        binary, contours, hir = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        thickness = 1
        if image_size > 256:
            thickness = 2
        ret = cv2.drawContours(image_color, contours, -1, (0, 255, 0), thickness)
        save_dir = args.save_dir + '/visContours' + str(image_size)
        mkdir(save_dir)
        cv2.imwrite(save_dir + '/' + str(cnt) + '.png', image_color)
        cnt += 1

        FPS.append(1 / (time1 - time0))
        Y.append(label_np)
        Y_pred.append(output_ins)

    cal_metrics = False
    if cal_metrics:
        fps = sum(FPS) / len(FPS)

        metric_names = ['criterion', 'thresh', 'fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 'n_true',
                        'n_pred', 'mean_true_score', 'mean_matched_score', 'panoptic_quality', 'by_image']

        stats = []
        taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for t in taus:
            m = matching_dataset(Y, Y_pred, thresh=t, show_progress=False)
            print(m)
            stats.append(m)

            metrics_dict = {}
            for i, info in enumerate(m):
                metrics_dict.setdefault(metric_names[i], info)
            metrics_dict.setdefault('fps', fps)
            with open(args.save_dir + '/test_metrics_iou_' + str(t) + '.json', 'w') as f:
                json.dump(metrics_dict, f, indent=2)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'):
            ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
        ax1.set_xlabel(r'IoU threshold $\tau$')
        ax1.set_ylabel('Metric value')
        ax1.grid()
        ax1.legend()
        for m in ('fp', 'tp', 'fn'):
            ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
        ax2.set_xlabel(r'IoU threshold $\tau$')
        ax2.set_ylabel('Number #')
        ax2.grid()
        ax2.legend()
        plt.savefig(args.save_dir + '/test_metrics.png')


if __name__ == "__main__":
    args = get_test_args()
    args.pt_root = 'Results'

    args.pt_dir = 'model1-unet-h256w256-erode0-weighting_none'
    args.test_images = '../../datasets/model1_test'
    args.save_dir = '../models/model1_unet'
    do_test(args)

    args.pt_dir = 'model2-unet-h256w256-erode0-weighting_none'
    args.test_images = '../../datasets/model2_test'
    args.save_dir = '../models/model2_unet'
    do_test(args)

    args.pt_dir = 'model3-unet-h256w256-erode0-weighting_none'
    args.test_images = '../../datasets/model3_test'
    args.save_dir = '../models/model3_unet'
    do_test(args)
