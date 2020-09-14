import os
import sys
from os.path import exists, join

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import utils.metrics as metrics
from config import config as configs
# from datasets import RegressionDataset as datasets
from models import model_pool as models
from utils.logger import MetricsLogger, CustomWriter


SAVE_ROOT = "training_checkpoints"


def dump_model(config, model, tag=None):
    save_dir = join(SAVE_ROOT, config.name)
    if not exists(save_dir):
        os.makedirs(save_dir)

    if tag:
        torch.save(model.state_dict(), join(save_dir, str(tag) + ".pth"))
    torch.save(model.state_dict(), join(save_dir, "latest.pth"))


def get_tensorboard_writer(config):
    save_dir = join(SAVE_ROOT, "tensorboard_logs", str(config.name))
    if not exists(save_dir):
        os.makedirs(save_dir)
    return CustomWriter(config, SummaryWriter(save_dir))


def dump_config(config):
    save_dir = join(SAVE_ROOT, config.name)
    if not exists(save_dir):
        os.makedirs(save_dir)
    filepath = join(save_dir, "{}.yaml".format(config.name))
    with open(filepath, "w") as f:
        print(config, file=f)


def get_config_dump(name):
    if "/" not in name:
        save_dir = join(SAVE_ROOT, name)
        filepath = join(save_dir, "{}.yaml".format(name))
    else:
        filepath = name
    config = configs.get_configuration(filepath)
    return config


def get_logger(config):
    """
    Returns logger.info function
    :param config:
    :return:
    """
    if config.system.stream == "file":
        sys.stdout = open(join(config.system.checkpoints_root, config.name, 'logs.txt'), 'a+')
        sys.stderr = sys.stdout

    return MetricsLogger(config)


def save_inference(config, phase, tag, x, y, y_hat):
    """Show image with landmarks for a batch of samples."""
    image_batch, target_batch = x, y
    output_batch = y_hat.cpu().detach().numpy()
    if isinstance(image_batch, type(torch.Tensor())):
        image_batch = image_batch.cpu().detach().numpy().transpose(0, 2, 3, 1)
        target_batch = target_batch.cpu().detach().numpy()
    image_batch = (image_batch - image_batch.min()) / (
            image_batch.max() - image_batch.min()
    )  # Adjust values for correct visualization
    # TODO: remove the need to manually change target scaling by adding it to CONFIG file
    # target_batch = (target_batch * 0.5 + 0.5) * config.dataset.img_size  # Denorm. of coordinates
    # output_batch = (output_batch * 0.5 + 0.5) * config.dataset.img_size  # Denorm. of coordinates
    rows = np.ceil(len(image_batch) / 4).astype(int)
    cols = 4
    fig, axs = plt.subplots(rows, cols, figsize=(35, 35))
    axs = axs.reshape(-1)
    for i, img in enumerate(image_batch):
        axs[i].imshow(img, vmin=0, vmax=1)
        for point in target_batch[i]:
            axs[i].scatter(point[0], point[1], c="r", s=2 ** 7)
        for point in output_batch[i]:
            axs[i].scatter(point[0], point[1], c="g", s=2 ** 7)
    # plt.show()
    save_dir = join(SAVE_ROOT, config.name, "visuals", phase)
    if not exists(save_dir):
        os.makedirs(save_dir)
    fig.savefig(join(save_dir, tag + ".png"))
    plt.close(fig)


def get_latest_epoch_num(config):
    l = os.listdir(join(SAVE_ROOT, config.name))
    l = [int(i[:-4]) for i in l if i.endswith(".pth") and i[:-4].isdigit()]
    return np.array(l).max().astype(int)


def get_all_predictions(model, device, data_loader):
    outputs = []
    targets = []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            data, target = batch_data["image"], batch_data["target"]
            data, target = data.to(device), target.to(device)
            output = model(data)
            outputs.append(output.cpu().detach().numpy())
            targets.append(target.cpu().detach().numpy())
    targets = np.concatenate(targets, axis=0).reshape(-1, 2)
    outputs = np.concatenate(outputs, axis=0).reshape(-1, 2)
    return targets, outputs


def plot_spatial_predictions(name):
    config = configs.get_configuration(os.path.join(SAVE_ROOT, name, name + ".yaml"))
    config.model["load_state"] = "latest"
    load = datasets.get_DataLoader(config, phase="test")
    model = models.get_model(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    targets, outputs = get_all_predictions(model, device, load)
    # targets = (targets * 0.5 + 0.5) * config.dataset.img_size  # Denorm. of coordinates
    # outputs = (outputs * 0.5 + 0.5) * config.dataset.img_size  # Denorm. of coordinates
    plt.axes().set_xlim([0, config.dataset.img_size])
    plt.axes().set_ylim([config.dataset.img_size, 0])
    plt.scatter(targets[:, 0], targets[:, 1], c="r", s=2 ** -1)
    plt.scatter(outputs[:, 0], outputs[:, 1], c="g", s=2 ** -1)
    save_path = os.path.join(SAVE_ROOT, config.name, "all_predictions.png")
    plt.gcf().savefig(os.path.join(save_path))
