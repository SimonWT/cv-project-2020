from yacs.config import CfgNode as CN

_C = CN()
_C.name = 'default_name'
_C.system = CN()
_C.system.num_gpu = 1
_C.system.num_workers = 4

_C.dataset = CN()
_C.dataset.img_size = 512
_C.dataset.augmentations = []
_C.dataset.transforms = []

_C.dataset.train = CN()
_C.dataset.train.csv_path = '/home/semyon/cardiomethry/ChestXrayIndex/train_small.csv'
_C.dataset.train.batch_size = 16
_C.dataset.train.shuffle = True

_C.dataset.val = CN()
_C.dataset.val.csv_path = '/home/semyon/cardiomethry/ChestXrayIndex/val.csv'
_C.dataset.val.batch_size = 16
_C.dataset.val.shuffle = False

_C.dataset.test = CN()
_C.dataset.test.csv_path = '/home/semyon/cardiomethry/ChestXrayIndex/test.csv'
_C.dataset.test.batch_size = 16
_C.dataset.test.shuffle = False


_C.model = CN()
_C.model.name = 'Baseline'
_C.model.in_channels = 3
_C.model.target_points = [0]
_C.model.final_activation = 'relu'
_C.model.weights_imagenet = False
_C.model.load_state = 0


_C.training = CN()
_C.training.num_epochs = 10
_C.training.dump_period = 5    # period(# of epochs) of saving model state
_C.training.log_interval = 20  # period(# of batches) to save visuals/log info
_C.training.optimizer = [{'lr': 0.001, 'name': 'RAdam'}]
_C.training.scheduler = [{'name': 'StepLR', 'step_size': 10}]

_C.training.criterion = []
_C.training.save_visuals = True

_C.testing = CN()
_C.testing.save_visuals = True

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
    cfg.freeze()
    return cfg

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        sys.argv.append('defaults.yaml')
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)