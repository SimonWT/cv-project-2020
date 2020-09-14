from pprint import pprint
from time import time

import torch

import datasets.HourglassHeatmapDataset as datasets
from datasets.HourglassHeatmapDataset import heatmap_to_coordinates
from utils import saver


def train_one_epoch(
        config,
        model,
        device,
        train_loader,
        optimizer,
        scheduler,  # TODO: make step, according do scheduler; LR to tensorboard
        criterion,
        epoch,
        logger,
        log_interval=20,
):
    model.train()
    train_loss = 0
    start_time = time()
    logger.reset()

    for batch_idx, batch_data in enumerate(train_loader):
        data, target, landmarks = batch_data["image"], batch_data["target"], batch_data["landmarks"]
        data, target, landmarks = data.to(device), target.to(device), landmarks.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        output = output[:, -1, ...]

        try:
            logger.add(landmarks, heatmap_to_coordinates(output), meta=batch_data["meta"])
        except BaseException:
            print(output.shape)
            raise NotImplementedError

        if batch_idx % log_interval == 0:
            if config.training.save_visuals:
                datasets.save_inference(
                    config,
                    "train",
                    batch_data,
                    output,
                    str(epoch) + "_" + str(batch_idx),
                )

        train_loss += loss.item()  # sum up batch loss

    train_loss /= len(train_loader)

    res = logger.calculate()
    print(
        "Train Epoch: {}/{} finished in {:.2f} min. \tLoss: {:.6f}".format(
            epoch,
            config.training.num_epochs + config.model.load_state,
            (time() - start_time) / 60,
            train_loss,
        )
    )
    pprint(res)

    # save model state
    if epoch % config.training.dump_period == 0:
        saver.dump_model(config, model, tag=epoch)

    res["loss"] = train_loss
    return res


def test(
        config,
        model,
        device,
        test_loader,
        criterion,
        logger,
        phase="val",
        tag="0",
        log_interval=8,
):
    logger.reset()
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            data, target, landmarks = (
                batch_data["image"],
                batch_data["target"],
                batch_data["landmarks"].float(),
            )
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()

            # resize target and output back to calculate metrics and save visuals
            output = output[:, -1, ...]  # Take only the last heatmap in hourglass stack

            output_coordinates = datasets.heatmap_to_coordinates(
                output
            )  # transform to coordinates

            logger.add(landmarks, output_coordinates, meta=batch_data["meta"], img_size=max(output.shape))
            log_interval = 100
            if config.testing.save_visuals and batch_idx % log_interval == 0:
                datasets.save_inference(
                    config, phase, batch_data, output, str(batch_idx) + "_" + str(tag)
                )

    test_loss /= len(test_loader)
    res = logger.calculate(phase=phase)
    pprint(res)
    print(
        "{} set after {}: Average loss: {:.4f}".format(
            phase.capitalize(), tag, test_loss
        )
    )
    res["loss"] = test_loss
    return res
