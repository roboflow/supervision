from __future__ import division, absolute_import
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import functional as F

__all__ = ['mobilenetv2_x1_0', 'mobilenetv2_x1_4']

model_urls = {
    # 1.0: top-1 71.3
    'mobilenetv2_x1_0':
    'https://mega.nz/#!NKp2wAIA!1NH1pbNzY_M2hVk_hdsxNM1NUOWvvGPHhaNr-fASF6c',
    # 1.4: top-1 73.9
    'mobilenetv2_x1_4':
    'https://mega.nz/#!RGhgEIwS!xN2s2ZdyqI6vQ3EwgmRXLEW3khr9tpXg96G9SUJugGk',
}


class ConvBlock(nn.Module):
    """Basic convolutional block.
    
    convolution (bias discarded) + batch normalization + relu6.

    Args:
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
        g (int): number of blocked connections from input channels
            to output channels (default: 1).
    """

    def __init__(self, in_c, out_c, k, s=1, p=0, g=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_c, out_c, k, stride=s, padding=p, bias=False, groups=g
        )
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu6(self.bn(self.conv(x)))


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, expansion_factor, stride=1):
        super(Bottleneck, self).__init__()
        mid_channels = in_channels * expansion_factor
        self.use_residual = stride == 1 and in_channels == out_channels
        self.conv1 = ConvBlock(in_channels, mid_channels, 1)
        self.dwconv2 = ConvBlock(
            mid_channels, mid_channels, 3, stride, 1, g=mid_channels
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        m = self.conv1(x)
        m = self.dwconv2(m)
        m = self.conv3(m)
        if self.use_residual:
            return x + m
        else:
            return m


class MobileNetV2(nn.Module):
    """MobileNetV2.

    Reference:
        Sandler et al. MobileNetV2: Inverted Residuals and
        Linear Bottlenecks. CVPR 2018.

    Public keys:
        - ``mobilenetv2_x1_0``: MobileNetV2 x1.0.
        - ``mobilenetv2_x1_4``: MobileNetV2 x1.4.
    """

    def __init__(
        self,
        num_classes,
        width_mult=1,
        loss='softmax',
        fc_dims=None,
        dropout_p=None,
        **kwargs
    ):
        super(MobileNetV2, self).__init__()
        self.loss = loss
        self.in_channels = int(32 * width_mult)
        self.feature_dim = int(1280 * width_mult) if width_mult > 1 else 1280

        # construct layers
        self.conv1 = ConvBlock(3, self.in_channels, 3, s=2, p=1)
        self.conv2 = self._make_layer(
            Bottleneck, 1, int(16 * width_mult), 1, 1
        )
        self.conv3 = self._make_layer(
            Bottleneck, 6, int(24 * width_mult), 2, 2
        )
        self.conv4 = self._make_layer(
            Bottleneck, 6, int(32 * width_mult), 3, 2
        )
        self.conv5 = self._make_layer(
            Bottleneck, 6, int(64 * width_mult), 4, 2
        )
        self.conv6 = self._make_layer(
            Bottleneck, 6, int(96 * width_mult), 3, 1
        )
        self.conv7 = self._make_layer(
            Bottleneck, 6, int(160 * width_mult), 3, 2
        )
        self.conv8 = self._make_layer(
            Bottleneck, 6, int(320 * width_mult), 1, 1
        )
        self.conv9 = ConvBlock(self.in_channels, self.feature_dim, 1)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(
            fc_dims, self.feature_dim, dropout_p
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._init_params()

    def _make_layer(self, block, t, c, n, s):
        # t: expansion factor
        # c: output channels
        # n: number of blocks
        # s: stride for first layer
        layers = []
        layers.append(block(self.in_channels, c, t, s))
        self.in_channels = c
        for i in range(1, n):
            layers.append(block(self.in_channels, c, t))
        return nn.Sequential(*layers)

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

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
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
            raise KeyError("Unsupported loss: {}".format(self.loss))


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


def mobilenetv2_x1_0(num_classes, loss, pretrained=True, **kwargs):
    model = MobileNetV2(
        num_classes,
        loss=loss,
        width_mult=1,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        # init_pretrained_weights(model, model_urls['mobilenetv2_x1_0'])
        import warnings
        warnings.warn(
            'The imagenet pretrained weights need to be manually downloaded from {}'
            .format(model_urls['mobilenetv2_x1_0'])
        )
    return model


def mobilenetv2_x1_4(num_classes, loss, pretrained=True, **kwargs):
    model = MobileNetV2(
        num_classes,
        loss=loss,
        width_mult=1.4,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        # init_pretrained_weights(model, model_urls['mobilenetv2_x1_4'])
        import warnings
        warnings.warn(
            'The imagenet pretrained weights need to be manually downloaded from {}'
            .format(model_urls['mobilenetv2_x1_4'])
        )
    return model
