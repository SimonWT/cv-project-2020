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