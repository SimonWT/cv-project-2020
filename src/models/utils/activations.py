from torch.nn import *


def get_loss(config):
    entry = config.model.final_activation[0]
    kwargs = {key: entry[key] for key in entry.keys() if key not in ['name']}
    if 'get_' + str(entry['name']) in globals():
        ret = globals()['get_' + str(entry['name'])](config, **kwargs)
    elif entry['name'] in globals():
        ret = globals()[entry['name']](**kwargs)
    else:
        raise NotImplementedError
    return ret

