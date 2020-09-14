import os
from os.path import join, exists

import albumentations as albu
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from datasets.utils.helpers import generate_target_heatmap
from datasets.utils.summurize_dataset import parse_json
from datasets.utils.transformsHeatmap import get_augmentations, get_transforms


class HeatmapDataset(Dataset):
    def __init__(self, config, phase="train"):
        self.df = pd.read_csv(config.dataset[phase].csv_path)
        dataset_dir = '/home/semyon/projects/cardiomethry/ChestXrayIndex'
        self.df.imgPath = self.df.imgPath.apply(lambda x: join(dataset_dir, x.split('ChestXrayIndex/')[-1]))
        self.df.annPath = self.df.annPath.apply(lambda x: join(dataset_dir, x.split('ChestXrayIndex/')[-1]))
        self.phase = phase
        self.transforms = get_transforms(config)
        self.augmentations = get_augmentations(config) if "train" in phase else None
        self.target_metrics = ["C", "CTI_A", "CTI_B", "MOOR_A", "LP_A", "LP_B"]
        self.config = config

        self.df = self.df.loc[:32]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        meta_data = self.df.loc[idx].to_dict()
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_path = self.df.iloc[idx, 1]
        target, _ = parse_json(ann_path)
        target = target.reshape(-1, 2)

        resize_pair = albu.Compose(
            [albu.Resize(self.config.dataset.img_size, self.config.dataset.img_size)],
            keypoint_params={"format": "xy", "remove_invisible": False},
        )
        augmented = resize_pair(image=image, keypoints=target)

        sample = {
            "image": augmented["image"],
            "target": np.around(np.array(augmented["keypoints"])),
        }

        sample["target"] = sample["target"][
            self.config.model.target_points
        ]  # sort out only needed points

        if self.augmentations and "train" in self.phase:
            augmented = self.augmentations(
                image=sample["image"], keypoints=sample["target"]
            )
            if (np.array(augmented["keypoints"]).min() < 0) or (
                    np.array(augmented["keypoints"]).max() > self.config.dataset.img_size
            ):
                print("Wrong augmentations")
            else:
                sample["image"] = augmented["image"]
                sample["target"] = np.array(augmented["keypoints"])

        # scale pixel values
        if self.config.dataset.scale:
            sample["image"] = (sample["image"] - sample["image"].min()) / (
                    sample["image"].max() - sample["image"].min()
            )
        sample["landmarks"] = sample["target"]

        # create target and landmarks matching the output of hourglass model
        target_heatmap_shape = (
            sample["image"].shape[0],
            sample["image"].shape[1],
            sample["image"].shape[-1],
        )
        # resize_pair = albu.Compose(
        #     [albu.Resize(sample["image"].shape[0] // 2, sample["image"].shape[0] // 2)],
        #     keypoint_params={"format": "xy", "remove_invisible": False},
        # )
        # augmented = resize_pair(image=sample["image"], keypoints=sample["landmarks"])

        # sample["landmarks"] = np.around(np.array(augmented["keypoints"]))
        sample["target"] = generate_target_heatmap(
            target_heatmap_shape, sample["landmarks"], sigma=self.config.dataset.sigma
        )

        # apply transformations
        if self.transforms:
            # Apply transform to numpy.ndarray which represents sample image
            transformed = self.transforms(
                image=sample["image"],
                target=sample["target"],
                keypoints=sample["landmarks"],
            )
            sample["image"] = transformed["image"]
            sample["landmarks"] = torch.from_numpy(transformed["keypoints"])
            sample["target"] = transformed["target"]

        sample["image"] = sample["image"].float()
        sample["target"] = sample["target"].float()
        sample["meta"] = meta_data
        return sample


def get_DataLoader(config, phase="train"):
    batch_size = config.dataset[phase].batch_size
    dataset = HeatmapDataset(config, phase=phase)
    shuffle = config.dataset[phase].shuffle
    num_workers = config.system.num_workers
    return DataLoader(dataset, batch_size, shuffle, num_workers=num_workers)


def show(image=None, target=None, output=None, landmarks=None, plot_point_width=3):
    """All variables must be of the same type, ndarray or torch.Tensor
    heatmap: (H,W, num_of_points)
    target: (num_of_points, 2)
    """

    empty_shape = (
        image.shape
        if image is not None
        else target.shape
        if target is not None
        else output
        if output is not None
        else (512, 512, 3)
    )
    if (
            isinstance(output, torch.Tensor)
            or isinstance(target, torch.Tensor)
            or isinstance(image, torch.Tensor)
            or isinstance(landmarks, torch.Tensor)
    ):
        new_axes = [0, 2, 3, 1] if len(empty_shape) == 4 else [1, 2, 0]
        if output is not None:
            output = output.cpu().detach().numpy().transpose(*new_axes)
        if target is not None:
            target = target.cpu().detach().numpy().transpose(*new_axes)
        if image is not None:
            image = image.cpu().detach().numpy().transpose(*new_axes)
        if landmarks is not None:
            landmarks = landmarks.cpu().detach().numpy()
    output_landmarks = None
    if output is not None:
        output_landmarks = heatmap_to_coordinates(output)

    empty_shape = (
        image.shape
        if image is not None
        else target.shape
        if target is not None
        else output
        if output is not None
        else (512, 512, 3)
    )

    if len(empty_shape) == 4:
        # print(image.shape, target.shape, output.shape)
        empty_shape = empty_shape[1:-1]
        n_in_row = 2
        rows = np.ceil(len(image) / n_in_row).astype(int)
        # print('axes: ', rows, 3 * n_in_row)
        fig, ax = plt.subplots(rows, 3 * n_in_row, sharey=True, figsize=(50, 70))
        # print(ax.shape)
        ax = ax.reshape(rows, 3 * n_in_row)
        # print(ax.shape)

        for i, _ in enumerate(image):
            offset = i % 2 * 3
            # input tile
            ax[i // 2 - i % 2, 0 + offset].imshow(
                np.zeros(empty_shape)
            ) if image is None else ax[i // 2 - i % 2, 0 + offset].imshow(image[i])

            # target tile
            ax[i // 2 - i % 2, 1 + offset].imshow(
                np.zeros(empty_shape)
            ) if target is None else ax[i // 2 - i % 2, 1 + offset].imshow(
                target[i].sum(axis=-1), vmin=0, vmax=1
            )

            # output tile
            ax[i // 2 - i % 2, 2 + offset].imshow(
                np.zeros(empty_shape)
            ) if output is None else ax[i // 2 - i % 2, 2 + offset].imshow(
                output[i].sum(axis=-1), vmin=0, vmax=1
            )
            if landmarks is not None:
                ax[i // 2 - i % 2, 0 + offset].scatter(
                    landmarks[i, :, 0],
                    landmarks[i, :, 1],
                    c="r",
                    s=2 ** plot_point_width,
                )
                ax[i // 2 - i % 2, 1 + offset].scatter(
                    landmarks[i, :, 0],
                    landmarks[i, :, 1],
                    c="r",
                    s=2 ** plot_point_width,
                )
                ax[i // 2 - i % 2, 2 + offset].scatter(
                    landmarks[i, :, 0],
                    landmarks[i, :, 1],
                    c="r",
                    s=2 ** plot_point_width,
                )
            if output_landmarks is not None:
                ax[i // 2 - i % 2, 0 + offset].scatter(
                    output_landmarks[i, :, 0],
                    output_landmarks[i, :, 1],
                    c="g",
                    s=2 ** plot_point_width,
                )
                ax[i // 2 - i % 2, 1 + offset].scatter(
                    output_landmarks[i, :, 0],
                    output_landmarks[i, :, 1],
                    c="g",
                    s=2 ** plot_point_width,
                )
                ax[i // 2 - i % 2, 2 + offset].scatter(
                    output_landmarks[i, :, 0],
                    output_landmarks[i, :, 1],
                    c="g",
                    s=2 ** plot_point_width,
                )
    else:
        empty_shape = empty_shape[:-1]
        fig, ax = plt.subplots(1, 3, sharey=True, figsize=(10, 30))
        ax[0].imshow(np.zeros(empty_shape)) if image is None else ax[0].imshow(image)
        ax[1].imshow(np.zeros(empty_shape)) if target is None else ax[1].imshow(
            target.sum(axis=-1), vmin=0, vmax=1
        )
        ax[2].imshow(np.zeros(empty_shape)) if output is None else ax[2].imshow(
            output.sum(axis=-1), vmin=0, vmax=1
        )

        if landmarks is not None:
            ax[0].scatter(
                landmarks[:, 0], landmarks[:, 1], c="r", s=2 ** plot_point_width
            )
            ax[1].scatter(
                landmarks[:, 0], landmarks[:, 1], c="r", s=2 ** plot_point_width
            )
            ax[2].scatter(
                landmarks[:, 0], landmarks[:, 1], c="r", s=2 ** plot_point_width
            )
    fig.savefig("tmp.png")
    return fig


def save_inference(config, phase, input_d, output, tag="default_tag"):
    """Show image with landmarks for a batch of samples."""
    data, target, landmarks = input_d["image"], input_d["target"], input_d["landmarks"]

    fig = show(data, target, output, landmarks)
    save_dir = join(config.system.checkpoints_root, config.name, "visuals", phase)
    if not exists(save_dir):
        os.makedirs(save_dir)
    fig.savefig(join(save_dir, tag + ".png"))
    plt.close(fig)


def heatmap_to_coordinates(heatmap_batch):
    """
    :param heatmap_batch: shape(batch_size, #_of_points, H,W)
    :return:
    """
    batch = []
    if isinstance(heatmap_batch, torch.Tensor):
        heatmap_batch = heatmap_batch.cpu().detach()
    elif isinstance(heatmap_batch, np.ndarray):
        heatmap_batch = heatmap_batch.transpose(0, 3, 1 ,2)
    for sample in heatmap_batch:
        landmarks = []
        for heatmap in sample:
            point = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            point = (point[1], point[0])
            point = np.array(point).reshape(-1, 2)
            landmarks.append(point)
        landmarks = np.concatenate(landmarks, axis=0)
        batch.append(landmarks)

    batch = np.stack(batch, axis=0)
    if isinstance(heatmap_batch, torch.Tensor):
        batch = torch.Tensor(batch)
    return batch


def upsample_heatmap(a, shape):
    out = []
    a = a.cpu().detach().numpy().transpose(0, 2, 3, 1)
    for hmap in a:
        #         print('hmap: ', hmap.shape)
        resized = cv2.resize(hmap, shape[-2:], interpolation=cv2.INTER_LINEAR)
        if len(resized.shape) == 2:
            resized = resized[..., np.newaxis]
        #         print('after: ', resized.shape)
        out.append(resized)
    out = np.stack(out)
    out = out.transpose(0, 3, 1, 2)
    ret = torch.Tensor(out)
    return ret
