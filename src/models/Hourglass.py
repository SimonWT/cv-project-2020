import torch
from torch import nn
import os
from collections import OrderedDict


def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(
            inp_dim,
            out_dim,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            bias=True,
        )
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Hourglass(nn.Module):
    def __init__(self, n, f, increase=0):
        """
        Autoencoder based on residuals blocks
        :param n: downsampling factor (n=3 means that 8x8 input becomes 1x1 in bottleneck)
        :param f: number of channels on input and output layers
        :param increase: num of channels to add to feature maps before recursive call
        """
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, f)
        # Lower branch
        self.pool1 = nn.MaxPool2d(2, 2)
        self.low1 = Residual(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n - 1, nf)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        #         if self.n == 1:
        #             print('Bottleneck batch shape: ', low2.shape)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class HourglassStack(nn.Module):
    def __init__(self, in_ch, nstack, inter_ch, out_ch, increase=0, **kwargs):
        """
        :param nstack: # of glasses
        :param inter_ch: # of channles on sides of hourglass
        :param out_ch: # of channels in target
        :param increase:
        :param kwargs:
        """
        super().__init__()

        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(in_ch, 64, 7, 1, bn=True, relu=True),  # not downsample twice
            Residual(64, 128),
            # nn.MaxPool2d(2, 2),  # downsample twice
            Residual(128, 128),
            Residual(128, inter_ch),
        )

        self.hgs = nn.ModuleList(
            [nn.Sequential(Hourglass(4, inter_ch, increase)) for _ in range(nstack)]
        )

        self.features = nn.ModuleList(
            [
                nn.Sequential(
                    Residual(inter_ch, inter_ch),
                    Conv(inter_ch, inter_ch, 1, bn=True, relu=True),
                )
                for _ in range(nstack)
            ]
        )

        # Transform hourglass output feature map to target via applying 1x1 convolution
        self.outs = nn.ModuleList(
            [Conv(inter_ch, out_ch, 1, relu=False, bn=False) for _ in range(nstack)]
        )
        self.merge_features = nn.ModuleList(
            [Merge(inter_ch, inter_ch) for _ in range(nstack - 1)]
        )
        self.merge_preds = nn.ModuleList(
            [Merge(out_ch, inter_ch) for _ in range(nstack - 1)]
        )
        self.nstack = nstack

    def forward(self, x):
        x = self.pre(x)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        res = torch.stack(combined_hm_preds, 1)
        return res


__all__ = ["get_Hourglass"]


def get_Hourglass(config, path_to_weights=None):
    if config.model.weights_imagenet:
        raise NotImplementedError
    net = HourglassStack(
        config.model.in_channels,
        config.model.hourglass_stack,
        config.model.hourglass_inter_channels,
        len(config.model.target_points),
        config.model.hourglass_inter_increase,
    )
    # net = nn.Sequential(net)

    if config.model.load_state == -1:
        config.model["load_state"] = "latest"
    if config.model.load_state:
        path_to_weights = os.path.join(
            config.system.checkpoints_root,
            config.name,
            str(config.model.load_state) + ".pth",
        )
        print("Restore the model from: {}".format(path_to_weights))

    if path_to_weights is None:

        def weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, torch.nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        print("Xavier init")
        net.apply(weights_init)
    else:
        state_dict = torch.load(path_to_weights)

        if "module" in list(state_dict.keys())[0]:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict

        net.load_state_dict(state_dict)
    return net
