import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


SAVE_ROOT = 'training_checkpoints'

def get_model(config):
    name = config.model.name
    model = globals()['get_'+name](config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)


def get_Baseline(config):
    net = Baseline(config=config)
    return net

class Baseline(nn.Module):
    def __init__(self, in_ch=3, config=None):
        super().__init__()
        n = 16
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, n, 3, padding=1),  # batch x 16 x 512 x 512
            nn.ReLU(),
            nn.BatchNorm2d(n),
            nn.Conv2d(n, n * 2, 3, padding=1),  # batch x 32 x 512 x 512
            nn.ReLU(),
            nn.BatchNorm2d(n * 2),
            nn.Conv2d(n * 2, n * 4, 3, padding=1),  # batch x 64 x 512 x 512
            nn.ReLU(),
            nn.BatchNorm2d(n * 4),
            nn.MaxPool2d(2, 2)  # batch x 64 x 256 x 256
        )
        self.layer2 = self.__get_layer__(n * 4, n * 8)  # batch x 128 x 128 x 128
        self.layer3 = self.__get_layer__(n * 8, n * 16)  # batch x 256 x 64 x 64
        self.layer4 = self.__get_layer__(n * 16, n * 32)  # batch x 512 x 32 x 32
        self.layer5 = nn.Sequential(
            nn.Conv2d(n * 32, 8, 5, 2),  # 16, 8, 14, 14
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Linear(8 * 14 * 14, 64),
            nn.Linear(64, 4),
            View(-1, len(config.model.target_points), 2),
            nn.Tanh()
        )

    def __get_layer__(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),  # batch x 16 x H x W
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
            nn.MaxPool2d(2, 2)  # batch x 64 x H/2 x W/2
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 8 * 14 * 14)
        x = self.layer6(x)
        return x

def get_Densenet121_1FC(config, path_to_weights=None):
    import re
    net = torchvision.models.densenet121()
    imagenet = config.model.weights_imagenet
    def get_classifier(num_f):
        return nn.Sequential(nn.Linear(num_f, len(config.model.target_points)*2 if config else 2),
                                        View(-1, len(config.model.target_points), 2),
                                        nn.Tanh()
                                        )

    if config.model.load_state == -1: config.model['load_state'] = 'latest'
    if config.model.load_state:
        imagenet = False
        path_to_weights = os.path.join(SAVE_ROOT, config.name, config.model.load_state+'.pth')
        print('Restore the model form: {}'.format(path_to_weights))
    if imagenet:
        state_dict = torch.load('/home/semyon/cardiomethry/src/models/weights/densenet121-a639ec97.pth')
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        net.load_state_dict(state_dict)
        num_ftrs = net.classifier.in_features
        net.classifier = get_classifier(num_ftrs)
    else:
        if path_to_weights == None:
            num_ftrs = net.classifier.in_features
            net.classifier = get_classifier(num_ftrs)

            def weights_init(m):
                if isinstance(m, nn.Conv2d) or isinstance(m, torch.nn.Linear):
                    nn.init.xavier_uniform_(m.weight.data)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
            print('Xavier init')
            net.apply(weights_init)
        else:
            state_dict = torch.load(path_to_weights)
            num_ftrs = net.classifier.in_features
            net.classifier = get_classifier(num_ftrs)
            net.load_state_dict(state_dict)


    return net

def get_Densenet121(config, imagenet=True, path_to_weights=None):
    import re
    net = torchvision.models.densenet121()
    if config.model.load_state:
        imagenet = False
        path_to_weights = os.path.join(SAVE_ROOT, config.name, config.model.load_state+'.pth')
        print('Restore the model form: {}'.format(path_to_weights))
    if imagenet:
        state_dict = torch.load('/home/semyon/cardiomethry/src/models/weights/densenet121-a639ec97.pth')
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        net.load_state_dict(state_dict)
        num_ftrs = net.classifier.in_features
        net.classifier = nn.Sequential(nn.Linear(num_ftrs, 512),
                                        nn.Linear(512, 32),
                                        nn.Linear(32, len(config.model.target_points)*2 if config else 2),
                                        View(-1, len(config.model.target_points), 2)
                                        )
    else:
        if path_to_weights == None:
            num_ftrs = net.classifier.in_features
            net.classifier = nn.Sequential(nn.Linear(num_ftrs, 512),
                                           nn.Linear(512, 32),
                                           nn.Linear(32, len(config.model.target_points) * 2 if config else 2),
                                           View(-1, len(config.model.target_points), 2)
                                           )
        else:
            state_dict = torch.load(path_to_weights)
            num_ftrs = net.classifier.in_features
            net.classifier = nn.Sequential(nn.Linear(num_ftrs, 512),
                                           nn.Linear(512, 32),
                                           nn.Linear(32, len(config.model.target_points) * 2 if config else 2),
                                           View(-1, len(config.model.target_points), 2)
                                           )
            net.load_state_dict(state_dict)
    return net

class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)

