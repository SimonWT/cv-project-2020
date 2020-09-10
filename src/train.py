import sys
from time import time

import torch
import torch.nn as nn
from datasets import RegressionDataset as datasets
from datasets.RegressionDataset import get_RegressionDataLoader
from models import model_pool as models
from tqdm import tqdm
from utils import saver
from utils import training_tools as tools

config = saver.get_config_dump(sys.argv[1])
logprint = saver.get_logger(config)

