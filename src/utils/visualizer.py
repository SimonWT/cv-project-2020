import numpy as np
import matplotlib.pyplot as plt
import datetime, pytz
import os
import torch


def show_batch(x, y, y_hat):
    """Show image with landmarks for a batch of samples.
    x: (batch_size, W, H, CH) or (batch_size, HC, W, H)"""
    image_batch, target_batch = x, y
    output_batch = y_hat
    if isinstance(image_batch, type(torch.Tensor())):
        image_batch = image_batch.numpy().transpose(0, 2, 3, 1)
        target_batch = target_batch.numpy()
    rows = np.ceil(len(image_batch) / 4).astype(int)
    cols = 4
    fig, axs = plt.subplots(rows, cols, figsize=(35, 35))
    axs = axs.reshape(-1)
    for i, img in enumerate(image_batch):
        axs[i].imshow(img)
        axs[i].scatter(target_batch[i][0], target_batch[i][1], c='r', s=2 ** 7)
        axs[i].scatter(output_batch[i][0], output_batch[i][1], c='r', s=2 ** 7)
    plt.show()


def show(image, target, output=None):
    """Show image with landmarks for a single sample."""
    if isinstance(image, type(torch.Tensor())):
        image = image.numpy().transpose(1, 2, 0)
        target = target.numpy()

    plt.imshow(image)
    print(target.shape)
    plt.scatter(target[:, 0], target[:, 1], c='r', s=2 ** 4)
    if output:
        plt.scatter(output[:][0], output[:][1], c='g', s=2 ** 4)
    plt.show()


def show_heatmap(heatmap, target):
    """Show heatmap with landmarks for a single sample.
    heatmap: (H,W, num_of_points)
    target: (num_of_points, 2)
    """
    if isinstance(heatmap, type(torch.Tensor())):
        heatmap = heatmap.numpy()#.transpose(1, 2, 0)
        target = target.numpy()
    print(heatmap.shape)
    plt.imshow(heatmap.sum(axis=-1), vmin=0, vmax=1)
    plt.scatter(target[:, 0], target[:, 1], c='r', s=2 ** 1)
    plt.show()
