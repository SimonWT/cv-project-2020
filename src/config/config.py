from yacs.config import CfgNode as CN
import os

_C = CN()
_C.name = "default_name"
_C.system = CN()
_C.system.num_gpu = 1
_C.system.num_workers = 4
_C.system.checkpoints_root = "training_checkpoints"
_C.system.stream = 'file'

_C.dataset = CN()
_C.dataset.img_size = 512
# _C.dataset.target_size = 256    # used for metrics computing
_C.dataset.augmentations = []
_C.dataset.scale = True
_C.dataset.sigma = 15
_C.dataset.transforms = [{"name": "ToTensorV2"}]
_C.dataset.root = "/home/semyon/cardiomethry/ChestXrayIndex"

_C.dataset.train = CN()
_C.dataset.train.csv_path = os.path.join(_C.dataset.root, "train.csv")
_C.dataset.train.batch_size = 16
_C.dataset.train.shuffle = True

_C.dataset.val = CN()
_C.dataset.val.csv_path = os.path.join(_C.dataset.root, "val.csv")
_C.dataset.val.batch_size = 16
_C.dataset.val.shuffle = False

_C.dataset.test = CN()
_C.dataset.test.csv_path = os.path.join(_C.dataset.root, "test.csv")
_C.dataset.test.batch_size = 16
_C.dataset.test.shuffle = False


_C.model = CN()
_C.model.name = "Unet"
_C.model.in_channels = 3
_C.model.target_points = [0]
_C.model.final_activation = [{"name": "ReLU"}]
_C.model.weights_imagenet = False
_C.model.weights_init = False
_C.model.load_state = 0

# Hourglass
_C.model.hourglass_stack = 2
_C.model.hourglass_inter_channels = 5
_C.model.hourglass_inter_increase = 4

_C.training = CN()
_C.training.num_epochs = 2
_C.training.dump_period = 5  # period(# of epochs) of saving model state
_C.training.log_interval = 20  # period(# of batches) to save visuals/log info
_C.training.optimizer = [{"lr": 0.001, "name": "RAdam"}]
_C.training.scheduler = [{"name": "StepLR", "step_size": 10}]
_C.training.criterion = [{"name": "MSELoss"}]
_C.training.save_visuals = True

_C.testing = CN()
_C.testing.save_visuals = True

# only the metrics defined in utils.metrics can be used
_C.testing.metrics = [{"name": "MAE"}, {"name": "PointwiseDist"}, {'name': 'PointwiseDistMM'}]


def __get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


def get_configuration(filename):
    """
    Obtain dict-like configuration object based on default configuration and updated
    with specified file.
    :param filename: path to .yaml config file
    :return: CfgNode object
    """
    cfg = __get_cfg_defaults()
    cfg.merge_from_file(filename)
    if cfg.model.load_state == "latest":
        cfg.model["load_state"] = -1
    cfg.freeze()
    return cfg


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        sys.argv.append("hourglass_test.yaml")
    print(_C)
    with open(sys.argv[1], "w") as f:
        print(_C, file=f)
