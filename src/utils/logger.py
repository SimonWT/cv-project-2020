from os.path import join

import pandas as pd
import torch

from datasets.utils.summurize_dataset import get_indices
from utils.metrics import get_metrics


class CustomWriter:
    def __init__(self, config, writer):
        self.config = config
        self.writer = writer

    def write_result(self, result_dict, epoch, phase):
        for name, value in result_dict.items():
            self.write_single(str(phase + "_" + name), value, epoch)

    def write_single(self, name, value, epoch):
        self.writer.add_scalar(self.config.name + "/" + name, value, epoch)

    def flush(self):
        self.writer.flush()


class MetricsLogger:
    def __init__(self, config):
        self.aggregated_true = []
        self.aggregated_pred = []
        self.meta = {}
        self.config = config
        self.funcs = get_metrics(config)
        self.num_points = len(config.model.target_points)
        self.img_size = 256

    def reset(self):
        self.__init__(self.config)

    def add(self, true, pred, **kwargs):
        assert isinstance(true, torch.Tensor), isinstance(pred, torch.Tensor)
        assert pred.shape == true.shape

        true = true.cpu().detach().double()
        pred = pred.cpu().detach().double()

        if 'img_size' in kwargs and kwargs['img_size'] != self.img_size:
            setattr(self, 'img_size', kwargs['img_size'])

        # stack metadata for all samples
        if 'meta' in kwargs:
            meta = kwargs['meta']
            if len(self.meta) == 0:
                self.meta = meta
            else:
                for key, value in meta.items():
                    if key not in self.meta:
                        continue
                    assert isinstance(self.meta[key], type(meta[key]))
                    if isinstance(meta[key], list):
                        self.meta[key] += meta[key]
                    elif isinstance(meta[key], torch.Tensor):
                        self.meta[key] = torch.cat([self.meta[key], meta[key]])
                    else:
                        print("Concatenation for {} not implemented".format(type(meta[key])))
                        raise NotImplementedError

        self.aggregated_true.append(true)
        self.aggregated_pred.append(pred)

    def calculate(self, phase='test'):
        self.aggregated_true = torch.cat(self.aggregated_true)
        self.aggregated_pred = torch.cat(self.aggregated_pred)

        x = {'x': self.aggregated_true, 'meta': self.meta, 'img_size': self.img_size}

        ret = {
            key: func(x,
                      self.aggregated_pred
                      )
            for key, func in self.funcs.items()
        }
        for key, value in self.meta.items():
            if isinstance(value, torch.Tensor):
                self.meta[key] = value.tolist()


        return {
            key: value.mean().item()
            for key, value in ret.items()
        }
