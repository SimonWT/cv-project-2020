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



def train_one_epoch(config, model, device, train_loader, optimizer, scheduler, criterion, epoch, log_interval=20):
    model.train()
    train_loss = 0
    start_time = time()
    for batch_idx, batch_data in enumerate(train_loader):
        data, target = batch_data['image'], batch_data['target']
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if batch_idx % log_interval == 0:
            if config.training.save_visuals:
                saver.save_inference(config, 'train', str(batch_idx) + '_' + str(epoch), data, target, output)

        train_loss += loss.item()  # sum up batch loss

    train_loss /= len(train_loader)

    logprint('Train Epoch: {}/{} finished in {:.2f} min. \tLoss: {:.6f}'.format(epoch,
                                                        config.training.num_epochs + config.model.load_state,
                                                        (time()-start_time)/60, train_loss))

    # save model state
    if epoch % config.training.dump_period == 0:
        saver.dump_model(config, model, tag=epoch)

    ret = {'loss': train_loss}
    return ret


def test(config, model, device, test_loader, criterion, phase='val', tag='0'):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            data, target = batch_data['image'], batch_data['target']
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()

            if config.testing.save_visuals:
                saver.save_inference(config, phase, str(batch_idx) + '_' + str(tag), data, target, output)

    test_loss /= len(test_loader)

    logprint('{} set after {}: Average loss: {:.4f}'.format(phase.capitalize(), tag, test_loss))
    ret = {'loss': test_loss}
    return ret


def run_training(config):
    num_epochs = config.training.num_epochs
    model = models.get_model(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader = datasets.get_RegressionDataLoader(config, phase='train')
    val_loader = datasets.get_RegressionDataLoader(config, phase='val')

    # optimizer = training_tools.RAdam(model.parameters(), lr=0.001, weight_decay=0.1)
    optimizer = tools.get_optimizer(model.parameters(), config)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = tools.get_scheduler(optimizer, config)
    #     scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=10)
    #     criterion = nn.PairwiseDistance()
    criterion = nn.MSELoss()

    log_writer = saver.get_tensorboard_writer(config)
    start_epoch_num = saver.get_latest_epoch_num(config) if config.model.load_state == -1 else config.model.load_state
    for epoch in tqdm(range(start_epoch_num + 1, start_epoch_num + num_epochs + 1)):
        print(epoch)
        train_buffer = train_one_epoch(config, model, device, train_loader, optimizer,
                                       scheduler, criterion, epoch, log_interval=config.training.log_interval)
        val_buffer = test(config, model, device, val_loader, criterion, tag=epoch)

        # plot the losses
        log_writer.add_scalars(config.name, {'train': train_buffer['loss'],
                                                'val': val_buffer['loss']}, epoch)
        log_writer.add_scalar(config.name+'/train_loss', train_buffer['loss'], epoch)
        log_writer.add_scalar(config.name+'/val_loss', val_buffer['loss'], epoch)
        log_writer.flush()

    test_loader = get_RegressionDataLoader(config, phase='test')
    test(config, model, device, test_loader, criterion, phase='test', tag= 'latest')

if __name__ == '__main__':
    run_training(config)

