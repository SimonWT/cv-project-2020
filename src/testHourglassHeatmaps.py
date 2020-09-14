import sys

import torch
import torch.nn as nn

from datasets import HourglassHeatmapDataset as datasets
from models import model_pool as models
from protocols.hourglass_heatmaps import test
from utils import saver
from utils import training_tools as tools


if __name__ == "__main__":
    config = saver.get_config_dump(sys.argv[1])
    config.system['stream'] = 'console'
    logger = saver.get_logger(config)

    config.model["load_state"] = sys.argv[2]
    model = models.get_model(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loader = datasets.get_DataLoader(config, phase="test")
    criterion = tools.get_loss(config)
    log_writer = saver.get_tensorboard_writer(config)
    print("work with: ", config.name)
    test_buffer = test(
        config,
        model,
        device,
        test_loader,
        criterion,
        logger,
        phase="test",
        tag=str(sys.argv[2]),
    )
