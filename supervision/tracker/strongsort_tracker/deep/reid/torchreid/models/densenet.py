"""
Code source: https://github.com/pytorch/vision
"""
from __future__ import division, absolute_import
import re
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils import model_zoo

__all__ = [
    'densenet121', 'densenet169', 'densenet201', 'densenet161',
    'densenet121_fc512'
]

model_urls = {
    'densenet121':
    'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169':
    'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201':
    'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161':
    'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module(
            'conv1',
            nn.Conv2d(
                num_input_features,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=False
            )
        ),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module(
            'conv2',
            nn.Conv2d(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
        ),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(
        self, num_layers, num_input_features, bn_size, growth_rate, drop_rate
    ):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i*growth_rate, growth_rate, bn_size,
                drop_rate
            )
            self.add_module('denselayer%d' % (i+1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False
            )
        )
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densely connected network.
    
    Reference:
        Huang et al. Densely Connected Convolutional Networks. CVPR 2017.

    Public keys:
        - ``densenet121``: DenseNet121.
        - ``densenet169``: DenseNet169.
        - ``densenet201``: DenseNet201.
        - ``densenet161``: DenseNet161.
        - ``densenet121_fc512``: DenseNet121 + FC.
    """

    def __init__(
        self,
        num_classes,
        loss,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        drop_rate=0,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    ):

        super(DenseNet, self).__init__()
        self.loss = loss

        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        'conv0',
                        nn.Conv2d(
                            3,
                            num_init_features,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False
                        )
                    ),
                    ('norm0', nn.BatchNorm2d(num_init_features)),
                    ('relu0', nn.ReLU(inplace=True)),
                    (
                        'pool0',
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                    ),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module('denseblock%d' % (i+1), block)
            num_features = num_features + num_layers*growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2
                )
                self.features.add_module('transition%d' % (i+1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = num_features
        self.fc = self._construct_fc_layer(fc_dims, num_features, dropout_p)

        # Linear layer
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._init_params()

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer.

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(
            fc_dims, (list, tuple)
        ), 'fc_dims must be either list or tuple, but got {}'.format(
            type(fc_dims)
        )

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        f = self.features(x)
        f = F.relu(f, inplace=True)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)

        if self.fc is not None:
            v = self.fc(v)

        if not self.training:
            return v

        y = self.classifier(v)

        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.
    
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)

    # '.'s are no longer allowed in module names, but pervious _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$'
    )
    for key in list(pretrain_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            pretrain_dict[new_key] = pretrain_dict[key]
            del pretrain_dict[key]

    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


"""
Dense network configurations:
--
densenet121: num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16)
densenet169: num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32)
densenet201: num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32)
densenet161: num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24)
"""


def densenet121(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = DenseNet(
        num_classes=num_classes,
        loss=loss,
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet121'])
    return model


def densenet169(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = DenseNet(
        num_classes=num_classes,
        loss=loss,
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 32, 32),
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet169'])
    return model


def densenet201(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = DenseNet(
        num_classes=num_classes,
        loss=loss,
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 48, 32),
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet201'])
    return model


def densenet161(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = DenseNet(
        num_classes=num_classes,
        loss=loss,
        num_init_features=96,
        growth_rate=48,
        block_config=(6, 12, 36, 24),
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet161'])
    return model


def densenet121_fc512(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = DenseNet(
        num_classes=num_classes,
        loss=loss,
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        fc_dims=[512],
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet121'])
    return model
