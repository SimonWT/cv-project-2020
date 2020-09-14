import random
import sys

import numpy as np
import torch
from tqdm import tqdm

from datasets import HourglassHeatmapDataset as datasets
from models import model_pool as models
from protocols.hourglass_heatmaps import train_one_epoch, test
from utils import saver
from utils import training_tools as tools


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_training(config):

    log_writer = saver.get_tensorboard_writer(config)
    logger = saver.get_logger(config)

    num_epochs = config.training.num_epochs
    model = models.get_model(config)
    model = torch.nn.DataParallel(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = datasets.get_DataLoader(config, phase="train")
    val_loader = datasets.get_DataLoader(config, phase="val")

    optimizer = tools.get_optimizer(model.parameters(), config)
    scheduler = tools.get_scheduler(optimizer, config)
    criterion = tools.get_loss(config)

    start_epoch_num = (
        saver.get_latest_epoch_num(config)
        if config.model.load_state == -1
        else config.model.load_state
    )
    for epoch in tqdm(range(start_epoch_num + 1, start_epoch_num + num_epochs + 1)):
        print("My#ep: ", epoch)
        train_buffer = train_one_epoch(
            config=config,
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            epoch=epoch,
            logger=logger,
            log_interval=config.training.log_interval,
        )

        val_buffer = test(
            config=config,
            model=model,
            device=device,
            test_loader=val_loader,
            criterion=criterion,
            logger=logger,
            phase="val",
            tag=epoch,
            log_interval=8,
        )

        log_writer.write_result(train_buffer, epoch, phase="train")
        log_writer.write_result(val_buffer, epoch, phase="val")


if __name__ == "__main__":
    set_seed(0)
    config = saver.get_config_dump(sys.argv[1])
    run_training(config)
