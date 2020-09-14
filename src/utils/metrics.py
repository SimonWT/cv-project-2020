from torch import nn
from torch.nn import *
import torch


def get_metrics(config):
    metrics_list = config.testing.metrics
    funcs = {}
    for entry in metrics_list:
        funcs[entry['name']] = __get_metric__(entry['name'])
    return funcs


def __get_metric__(name):
    if 'get_' + str(name) in globals():
        ret = globals()['get_' + name]()
    elif name in globals():
        ret = globals()[name]()
    else:
        raise NotImplementedError
    return ret


class PointwiseDist(nn.Module):
    """
    Distance between points in space in px
    param: x: {'x': input_array, ...}
    Inputs: shape (batch_size, #_of_points,  #_of_axes)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = x['x']
        return x.sub(y).pow(2).sum(axis=-1).pow(1 / 2).mean(-1)




class MAE(nn.Module):
    """
    Inputs: shape (batch_size, #_of_points,  #_of_axes)
            [x,y]
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = x['x']
        f = L1Loss(reduction='none')
        return f(x, y).mean(-1).mean(-1)