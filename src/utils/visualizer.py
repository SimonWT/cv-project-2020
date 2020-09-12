import numpy as np
import matplotlib.pyplot as plt
import datetime, pytz
import os
import torch

def show_batch(x, y, y_hat):
    """Show image with landmarks for a batch of samples."""
    image_batch, target_batch = x, y
    output_batch = y_hat
    if isinstance(image_batch, type(torch.Tensor())):
        image_batch = image_batch.numpy().transpose(0,2,3,1)
        target_batch = target_batch.numpy()
    rows = np.ceil(len(image_batch)/4).astype(int)
    cols = 4
    fig, axs = plt.subplots(rows, cols, figsize=(35,35))
    axs = axs.reshape(-1)
    for i, img in enumerate(image_batch):
        axs[i].imshow(img)
        axs[i].scatter(target_batch[i][0], target_batch[i][1], c='r', s=2**7)
        axs[i].scatter(output_batch[i][0], output_batch[i][1], c='r', s=2 ** 7)
    plt.show()


