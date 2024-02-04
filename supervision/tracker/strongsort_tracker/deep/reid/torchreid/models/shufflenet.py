from __future__ import division, absolute_import
import torch
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import functional as F

__all__ = ['shufflenet']

model_urls = {
    # training epoch = 90, top1 = 61.8
    'imagenet':
    'https://mega.nz/#!RDpUlQCY!tr_5xBEkelzDjveIYBBcGcovNCOrgfiJO9kiidz9fZM',
}


class ChannelShuffle(nn.Module):

    def __init__(self, num_groups):
        super(ChannelShuffle, self).__init__()
        self.g = num_groups

    def forward(self, x):
        b, c, h, w = x.size()
        n = c // self.g
        # reshape
        x = x.view(b, self.g, n, h, w)
        # transpose
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # flatten
        x = x.view(b, c, h, w)
        return x


class Bottleneck(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        num_groups,
        group_conv1x1=True
    ):
        super(Bottleneck, self).__init__()
        assert stride in [1, 2], 'Warning: stride must be either 1 or 2'
        self.stride = stride
        mid_channels = out_channels // 4
        if stride == 2:
            out_channels -= in_channels
        # group conv is not applied to first conv1x1 at stage 2
        num_groups_conv1x1 = num_groups if group_conv1x1 else 1
        self.conv1 = nn.Conv2d(
            in_channels,
            mid_channels,
            1,
            groups=num_groups_conv1x1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.shuffle1 = ChannelShuffle(num_groups)
        self.conv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            3,
            stride=stride,
            padding=1,
            groups=mid_channels,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(
            mid_channels, out_channels, 1, groups=num_groups, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        if stride == 2:
            self.shortcut = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        if self.stride == 2:
            res = self.shortcut(x)
            out = F.relu(torch.cat([res, out], 1))
        else:
            out = F.relu(x + out)
        return out


# configuration of (num_groups: #out_channels) based on Table 1 in the paper
cfg = {
    1: [144, 288, 576],
    2: [200, 400, 800],
    3: [240, 480, 960],
    4: [272, 544, 1088],
    8: [384, 768, 1536],
}


class ShuffleNet(nn.Module):
    """ShuffleNet.

    Reference:
        Zhang et al. ShuffleNet: An Extremely Efficient Convolutional Neural
        Network for Mobile Devices. CVPR 2018.

    Public keys:
        - ``shufflenet``: ShuffleNet (groups=3).
    """

    def __init__(self, num_classes, loss='softmax', num_groups=3, **kwargs):
        super(ShuffleNet, self).__init__()
        self.loss = loss

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.stage2 = nn.Sequential(
            Bottleneck(
                24, cfg[num_groups][0], 2, num_groups, group_conv1x1=False
            ),
            Bottleneck(cfg[num_groups][0], cfg[num_groups][0], 1, num_groups),
            Bottleneck(cfg[num_groups][0], cfg[num_groups][0], 1, num_groups),
            Bottleneck(cfg[num_groups][0], cfg[num_groups][0], 1, num_groups),
        )

        self.stage3 = nn.Sequential(
            Bottleneck(cfg[num_groups][0], cfg[num_groups][1], 2, num_groups),
            Bottleneck(cfg[num_groups][1], cfg[num_groups][1], 1, num_groups),
            Bottleneck(cfg[num_groups][1], cfg[num_groups][1], 1, num_groups),
            Bottleneck(cfg[num_groups][1], cfg[num_groups][1], 1, num_groups),
            Bottleneck(cfg[num_groups][1], cfg[num_groups][1], 1, num_groups),
            Bottleneck(cfg[num_groups][1], cfg[num_groups][1], 1, num_groups),
            Bottleneck(cfg[num_groups][1], cfg[num_groups][1], 1, num_groups),
            Bottleneck(cfg[num_groups][1], cfg[num_groups][1], 1, num_groups),
        )

        self.stage4 = nn.Sequential(
            Bottleneck(cfg[num_groups][1], cfg[num_groups][2], 2, num_groups),
            Bottleneck(cfg[num_groups][2], cfg[num_groups][2], 1, num_groups),
            Bottleneck(cfg[num_groups][2], cfg[num_groups][2], 1, num_groups),
            Bottleneck(cfg[num_groups][2], cfg[num_groups][2], 1, num_groups),
        )

        self.classifier = nn.Linear(cfg[num_groups][2], num_classes)
        self.feat_dim = cfg[num_groups][2]

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)

        if not self.training:
            return x

        y = self.classifier(x)

        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, x
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.
    
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


def shufflenet(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ShuffleNet(num_classes, loss, **kwargs)
    if pretrained:
        # init_pretrained_weights(model, model_urls['imagenet'])
        import warnings
        warnings.warn(
            'The imagenet pretrained weights need to be manually downloaded from {}'
            .format(model_urls['imagenet'])
        )
    return model
