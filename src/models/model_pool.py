import torch
import torch.nn as nn

from models.Hourglass import *


SAVE_ROOT = 'training_checkpoints'

def get_model(config):
    """
    1) define arch.
    2) append final act.
    3) a) load checkpoint
        b) init params.
    4) .to(dev.)
    :param config:
    :return:
    """
    name = config.model.name
    model = globals()['get_'+name](config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)

