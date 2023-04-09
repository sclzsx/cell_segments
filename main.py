import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from csbdeep.utils import normalize
from PIL import Image
from pathlib import Path
from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available, _draw_polygons
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D
import json


def plot_img_label(save_plt_path, img, lbl, img_title="image", lbl_title="label", **kwargs):
    lbl_cmap = random_label_cmap()
    fig, (ai, al) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw=dict(width_ratios=(1.25, 1)))
    im = ai.imshow(img, cmap='gray', clim=(0, 1))
    ai.set_title(img_title)
    fig.colorbar(im, ax=ai)
    al.imshow(lbl, cmap=lbl_cmap)
    al.set_title(lbl_title)
    plt.tight_layout()
    # plt.axes('off')
    plt.savefig(save_plt_path)


def random_fliprot(img, mask):
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(perm)
    for ax in axes:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask


def random_intensity_change(img):
    img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
    return img


def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02 * np.random.uniform(0, 1)
    x = x + sig * np.random.normal(0, 1, x.shape)
    return x, y


def load_dataset(mask_dir):
    X, Y = [], []
    for mask_path in Path(mask_dir).glob('*_mask.png'):
        mask_path = str(mask_path)
        image_path = mask_path.replace('_mask', '')
        image = Image.open(image_path)
        image = image.convert('L')
        image = image.resize((256, 256))
        image = np.array(image)
        mask = Image.open(mask_path)
        mask = mask.resize((256, 256), Image.NEAREST)
        mask = np.array(mask)
        X.append(image)
        Y.append(mask)
    check = False
    if check:
        x1, y1 = X[0], Y[0]
        print(x1.shape, x1.dtype, np.min(x1), np.max(x1), type(x1))
        print(y1.shape, y1.dtype, np.min(y1), np.max(y1), type(y1))
        print(Counter(x1.flatten()))
        print(Counter(y1.flatten()))
        K = []
        for k, v in Counter(y1.flatten()).items():
            K.append(k)
        K.sort()
        print(K)
        plt.imshow(y1)
        plt.show()
    return X, Y


def preprocess_dataset(X, Y):
    axis_norm = (0, 1)  # normalize channels independently
    # axis_norm = (0, 1, 2)  # normalize channels jointly
    X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X)]
    Y = [fill_label_holes(y) for y in tqdm(Y)]
    return X, Y


def split_dataset_train_val(X, Y):
    assert len(X) > 1, "not enough training data"
    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(0.15 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
    print('number of images: %3d' % len(X))
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))
    return X_trn, Y_trn, X_val, Y_val


def train_val(mask_dir, save_name, save_root, with_dist_loss=True):
    X, Y = load_dataset(mask_dir)
    X, Y = preprocess_dataset(X, Y)
    X_trn, Y_trn, X_val, Y_val = split_dataset_train_val(X, Y)

    n_channel = 1 if X_trn[0].ndim == 2 else X_trn[0].shape[-1]

    # 32 is a good default choice (see 1_data.ipynb)
    n_rays = 32

    # Use OpenCL-based computations for data generator during training (requires 'gputools')
    use_gpu = False and gputools_available()
    print('use_gpu', use_gpu)

    # Predict on subsampled grid for increased efficiency and larger field of view
    grid = (2, 2)

    if with_dist_loss:
        conf = Config2D(
            n_rays=n_rays,
            grid=grid,
            use_gpu=use_gpu,
            n_channel_in=n_channel,
        )
    else:
        conf = Config2D(
            n_rays=n_rays,
            grid=grid,
            use_gpu=use_gpu,
            n_channel_in=n_channel,
            train_loss_weights=(1, 0),
        )
    print(conf)
    vars(conf)

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory

        # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
        limit_gpu_memory(0.8)
        # alternatively, try this:
        # limit_gpu_memory(None, allow_growth=True)

    model = StarDist2D(conf, name=save_name, basedir=save_root)

    median_size = calculate_extents(list(Y_trn), np.median)
    fov = np.array(model._axes_tile_overlap('YX'))
    print(f"median object size:      {median_size}")
    print(f"network field of view :  {fov}")
    if any(median_size > fov):
        print("WARNING: median object size larger than field of view of the neural network.")

    history = model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter)

    model.optimize_thresholds(X_val, Y_val)

    Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
                  for x in tqdm(X_val)]

    save_plt_dir = save_root + '/' + save_name

    plot_img_label(save_plt_dir + '/val_gt.png', X_val[0], Y_val[0], lbl_title="label GT")
    plot_img_label(save_plt_dir + '/val_pred.png', X_val[0], Y_val_pred[0], lbl_title="label Pred")

    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    stats = [matching_dataset(Y_val, Y_val_pred, thresh=t, show_progress=False) for t in tqdm(taus)]

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

    plt.savefig(save_plt_dir + '/val_metrics.png')


def do_test(mask_dir, save_name, save_root='./models', with_dist_loss=True):
    X_val, Y_val = load_dataset(mask_dir)
    X_val, Y_val = preprocess_dataset(X_val, Y_val)

    n_channel = 1 if X_val[0].ndim == 2 else Y_val[0].shape[-1]

    # 32 is a good default choice (see 1_data.ipynb)
    n_rays = 32

    # Predict on subsampled grid for increased efficiency and larger field of view
    grid = (2, 2)

    if with_dist_loss:
        conf = Config2D(
            n_rays=n_rays,
            grid=grid,
            use_gpu=False,
            n_channel_in=n_channel,
        )
    else:
        conf = Config2D(
            n_rays=n_rays,
            grid=grid,
            use_gpu=False,
            n_channel_in=n_channel,
            train_loss_weights=(1, 0),
        )
    print(conf)
    vars(conf)

    model = StarDist2D(conf, name=save_name, basedir=save_root)
    model.load_weights()

    save_plt_dir = save_root + '/' + save_name

    Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
                  for x in tqdm(X_val)]
    metrics_05 = matching_dataset(Y_val, Y_val_pred, thresh=0.5, show_progress=False)
    print(metrics_05)
    metric_names = ['criterion', 'thresh', 'fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 'n_true',
                    'n_pred', 'mean_true_score', 'mean_matched_score', 'panoptic_quality', 'by_image']
    metrics_05_dict = {}
    for i, info in enumerate(metrics_05):
        metrics_05_dict.setdefault(metric_names[i], info)
    with open(save_plt_dir + '/test_metrics_iou05.json', 'w') as f:
        json.dump(metrics_05_dict, f, indent=2)

    test_metric_curves = True
    if test_metric_curves:
        Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
                      for x in tqdm(X_val)]

        plot_img_label(save_plt_dir + '/test_gt.png', X_val[0], Y_val[0], lbl_title="label GT")
        plot_img_label(save_plt_dir + '/test_pred.png', X_val[0], Y_val_pred[0], lbl_title="label Pred")

        taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        stats = [matching_dataset(Y_val, Y_val_pred, thresh=t, show_progress=False) for t in tqdm(taus)]

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

        plt.savefig(save_plt_dir + '/test_metrics.png')

    sample_step = int(0.25 * len(X_val))
    for i in range(0, len(X_val), sample_step):
        save_test_plt(save_plt_dir + '/test_sample_' + str(i) + '.png', model, X_val[i])


def save_test_plt(save_plt_path, model, x, show_dist=True):
    labels, details = model.predict_instances(x)
    lbl_cmap = random_label_cmap()
    plt.figure(figsize=(13, 10))
    img_show = x if x.ndim == 2 else x[..., 0]
    coord, points, prob = details['coord'], details['points'], details['prob']
    plt.subplot(121)
    plt.imshow(img_show, cmap='gray')
    plt.axis('off')
    a = plt.axis()
    _draw_polygons(coord, points, prob, show_dist=show_dist)
    plt.axis(a)
    plt.subplot(122)
    plt.imshow(img_show, cmap='gray')
    plt.axis('off')
    plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_plt_path)
    print('saved to', save_plt_path)


if __name__ == '__main__':
    np.random.seed(42)

    # save_name = 'model1_stardist'
    # mask_dir_train = '../datasets/model1_training/all_straight'
    # mask_dir_test = '../datasets/model1_test'
    # train_val(mask_dir_train, save_name, save_root='./models')
    # do_test(mask_dir_test, save_name, save_root='./models')

    # save_name = 'model2_stardist'
    # mask_dir_train = '../datasets/model2_training'
    # mask_dir_test = '../datasets/model2_test'
    # train_val(mask_dir_train, save_name, save_root='./models')
    # do_test(mask_dir_test, save_name, save_root='./models')

    # save_name = 'model3_stardist'
    # mask_dir_train = '../datasets/model3_training'
    # mask_dir_test = '../datasets/model3_test'
    # train_val(mask_dir_train, save_name, save_root='./models')
    # do_test(mask_dir_test, save_name, save_root='./models')

    # save_name = 'model2_unet'
    # mask_dir_train = '../datasets/model2_training'
    # mask_dir_test = '../datasets/model2_test'
    # train_val(mask_dir_train, save_name, save_root='./models', with_dist_loss=False)
    # do_test(mask_dir_test, save_name, save_root='./models', with_dist_loss=False)
