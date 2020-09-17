from torch.nn import *
import torch


class HourglassLoss(Module):
    """
    MSE averaged over all prediction maps
    """

    def __init__(self):
        super().__init__()

    def batch_loss(self, output, target):
        # MSE
        return ((output - target)**2).mean() #TODO: add any loss type

    def forward(self, combined_heatmap_output, target):
        combined_loss = []
        for intermediate_heatmap in combined_heatmap_output.permute(1,0,2,3,4):
            combined_loss.append(self.batch_loss(intermediate_heatmap, target))
        combined_loss = torch.stack(combined_loss, dim=0)
        return combined_loss.mean(dim=0)


def get_loss(config):
    entry = config.training.criterion[0]
    kwargs = {key: entry[key] for key in entry.keys() if key not in ['name']}
    if 'get_' + str(entry['name']) in globals():
        ret = globals()['get_' + str(entry['name'])](config, **kwargs)
    elif entry['name'] in globals():
        ret = globals()[entry['name']](**kwargs)
    else:
        raise NotImplementedError
    return ret



