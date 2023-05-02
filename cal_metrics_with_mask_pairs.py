from pathlib import Path
from tqdm import tqdm
from PIL import Image
import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
from stardist.matching import matching_dataset
import json


def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def process_results(pred_dir, label_dir, reSize, save_dir, cal_metrics):
    mkdir(save_dir)
    Y, Y_pred, = [], []
    label_paths = [i for i in Path(label_dir).glob('*_mask.png')]
    cnt = 0
    for label_path in tqdm(label_paths):
        label_name = label_path.name
        pred_name = label_name.replace('_mask', '')
        pred_path = pred_dir + '/' + pred_name
        label_path = str(label_path)
        img_path = label_path.replace('_mask', '')

        if not (os.path.exists(img_path) and os.path.exists(pred_path) and os.path.exists(label_path)):
            continue
        appoint_size = (reSize, reSize)

        pred = Image.open(pred_path)
        pred = pred.resize(appoint_size, Image.NEAREST)
        pred_np = np.array(pred)
        output = np.where(pred_np > 0, 255, 0).astype('uint8')

        image_color = Image.open(img_path)
        image = image_color.convert('L')
        label = Image.open(label_path)
        image = image.resize(appoint_size)
        label = label.resize(appoint_size, Image.NEAREST)
        label_np = np.array(label)

        image_color = image_color.resize(appoint_size)
        image_color = np.array(image_color)
        image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
        binary, contours, hir = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        ret = cv2.drawContours(image_color, contours, -1, (0, 255, 0), 2)
        save_vis_dir = save_dir + '/visContours' + str(reSize)
        mkdir(save_vis_dir)

        cv2.imwrite(save_vis_dir + '/' + str(cnt) + '.png', image_color)
        cnt += 1

        Y.append(label_np)
        Y_pred.append(pred_np)

    if cal_metrics:
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
            with open(save_dir + '/test_metrics_iou_' + str(t) + '.json', 'w') as f:
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
        plt.savefig(save_dir + '/test_metrics.png')


def process_results_single_mask_pair(pred_path, label_path, reSize):
    img_path = label_path.replace('_mask', '')

    if not (os.path.exists(img_path) and os.path.exists(pred_path) and os.path.exists(label_path)):
        return None

    appoint_size = (reSize, reSize)

    pred = Image.open(pred_path)
    pred = pred.resize(appoint_size, Image.NEAREST)
    pred_np = np.array(pred)

    label = Image.open(label_path)
    label = label.resize(appoint_size, Image.NEAREST)
    label_np = np.array(label)

    return (label_np, pred_np)


def cal_metrics_single_mask_pair(Y, Y_pred, save_dir):
    mkdir(save_dir)

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
        with open(save_dir + '/test_metrics_iou_' + str(t) + '.json', 'w') as f:
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
    plt.savefig(save_dir + '/test_metrics.png')


def cal_metrics_2_methods(pred_dir1, pred_dir2, label_dir, reSize, save_dir1, save_dir2):
    Y, Y_pred1, Y_pred2 = [], [], []
    label_paths = [i for i in Path(label_dir).glob('*_mask.png')]

    for label_path in tqdm(label_paths):
        label_name = label_path.name
        pred_name = label_name.replace('_mask', '')
        pred_path1 = pred_dir1 + '/' + pred_name
        pred_path2 = pred_dir2 + '/' + pred_name
        label_path = str(label_path)

        ret1 = process_results_single_mask_pair(pred_path1, label_path, reSize)
        ret2 = process_results_single_mask_pair(pred_path2, label_path, reSize)

        if ret1 is not None and ret2 is not None:
            Y.append(ret1[0])
            Y_pred1.append(ret1[1])
            Y_pred2.append(ret2[1])

    cal_metrics_single_mask_pair(Y, Y_pred1, save_dir1)
    cal_metrics_single_mask_pair(Y, Y_pred1, save_dir2)


if __name__ == '__main__':
    # pred_dir = 'maskRCNNpreds/model_1'
    # label_dir = '../datasets/model1_test'
    # reSize = 512
    # save_dir = 'maskRCNNpreds/model_1_metrics'
    # cal_metrics = True
    # process_results(pred_dir, label_dir, reSize, save_dir, cal_metrics)
    #
    # pred_dir = 'maskRCNNpreds/model_2'
    # label_dir = '../datasets/model2_test'
    # reSize = 512
    # save_dir = 'maskRCNNpreds/model_2_metrics'
    # cal_metrics = True
    # process_results(pred_dir, label_dir, reSize, save_dir, cal_metrics)
    #
    # pred_dir = 'maskRCNNpreds/model_3'
    # label_dir = '../datasets/model3_test'
    # reSize = 512
    # save_dir = 'maskRCNNpreds/model_3_metrics'
    # cal_metrics = True
    # process_results(pred_dir, label_dir, reSize, save_dir, cal_metrics)

    ######## extend

    # pred_dir = '../extend/maskRCNNpreds/model_1_duck_20'
    # label_dir = '../extend/testsets/model_1_all_test_duck_20'
    # reSize = 512
    # save_dir = '../extend/maskRCNNpredsMetrics/model_1_all_test_duck_20'
    # cal_metrics = True
    # process_results(pred_dir, label_dir, reSize, save_dir, cal_metrics)
    #
    # pred_dir = '../extend/maskRCNNpreds/model_2_duck_50'
    # label_dir = '../extend/testsets/model_2_all_test_duck_50'
    # reSize = 512
    # save_dir = '../extend/maskRCNNpredsMetrics/model_2_all_test_duck_50'
    # cal_metrics = True
    # process_results(pred_dir, label_dir, reSize, save_dir, cal_metrics)
    #
    # pred_dir = '../extend/maskRCNNpreds/model_2_pig_20'
    # label_dir = '../extend/testsets/model_2_all_test-pig_20'
    # reSize = 512
    # save_dir = '../extend/maskRCNNpredsMetrics/model_2_all_test-pig_20'
    # cal_metrics = True
    # process_results(pred_dir, label_dir, reSize, save_dir, cal_metrics)
    #
    # pred_dir = '../extend/maskRCNNpreds/model_3_cattle_35'
    # label_dir = '../extend/testsets/model_3_all_test_cattle_35'
    # reSize = 512
    # save_dir = '../extend/maskRCNNpredsMetrics/model_3_all_test_cattle_35'
    # cal_metrics = True
    # process_results(pred_dir, label_dir, reSize, save_dir, cal_metrics)
    #
    # pred_dir = '../extend/maskRCNNpreds/model_3_mice_100'
    # label_dir = '../extend/testsets/model_3_all_test_mice_100'
    # reSize = 512
    # save_dir = '../extend/maskRCNNpredsMetrics/model_3_all_test_mice_100'
    # cal_metrics = True
    # process_results(pred_dir, label_dir, reSize, save_dir, cal_metrics)
    #
    # pred_dir = '../extend/maskRCNNpreds/model_3_pig_50'
    # label_dir = '../extend/testsets/model_3_all_test_pig_50'
    # reSize = 512
    # save_dir = '../extend/maskRCNNpredsMetrics/model_3_all_test_pig_50'
    # cal_metrics = True
    # process_results(pred_dir, label_dir, reSize, save_dir, cal_metrics)

    ############ check 0423

    cal_metrics_2_methods(pred_dir1, pred_dir2, label_dir, reSize, save_dir1, save_dir2)

    pred_dir1 = 'maskRCNNpreds/model_1'
    label_dir = '../datasets/model1_test'
    reSize = 512
    save_dir = 'maskRCNNpreds/model_1_metrics'

