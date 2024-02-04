from __future__ import division, absolute_import
import torch
from torch import nn
from torch.nn import functional as F

__all__ = ['HACNN']


class ConvBlock(nn.Module):
    """Basic convolutional block.
    
    convolution + batch normalization + relu.

    Args:
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
    """

    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class InceptionA(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(InceptionA, self).__init__()
        mid_channels = out_channels // 4

        self.stream1 = nn.Sequential(
            ConvBlock(in_channels, mid_channels, 1),
            ConvBlock(mid_channels, mid_channels, 3, p=1),
        )
        self.stream2 = nn.Sequential(
            ConvBlock(in_channels, mid_channels, 1),
            ConvBlock(mid_channels, mid_channels, 3, p=1),
        )
        self.stream3 = nn.Sequential(
            ConvBlock(in_channels, mid_channels, 1),
            ConvBlock(mid_channels, mid_channels, 3, p=1),
        )
        self.stream4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBlock(in_channels, mid_channels, 1),
        )

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        s4 = self.stream4(x)
        y = torch.cat([s1, s2, s3, s4], dim=1)
        return y


class InceptionB(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(InceptionB, self).__init__()
        mid_channels = out_channels // 4

        self.stream1 = nn.Sequential(
            ConvBlock(in_channels, mid_channels, 1),
            ConvBlock(mid_channels, mid_channels, 3, s=2, p=1),
        )
        self.stream2 = nn.Sequential(
            ConvBlock(in_channels, mid_channels, 1),
            ConvBlock(mid_channels, mid_channels, 3, p=1),
            ConvBlock(mid_channels, mid_channels, 3, s=2, p=1),
        )
        self.stream3 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            ConvBlock(in_channels, mid_channels * 2, 1),
        )

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        y = torch.cat([s1, s2, s3], dim=1)
        return y


class SpatialAttn(nn.Module):
    """Spatial Attention (Sec. 3.1.I.1)"""

    def __init__(self):
        super(SpatialAttn, self).__init__()
        self.conv1 = ConvBlock(1, 1, 3, s=2, p=1)
        self.conv2 = ConvBlock(1, 1, 1)

    def forward(self, x):
        # global cross-channel averaging
        x = x.mean(1, keepdim=True)
        # 3-by-3 conv
        x = self.conv1(x)
        # bilinear resizing
        x = F.upsample(
            x, (x.size(2) * 2, x.size(3) * 2),
            mode='bilinear',
            align_corners=True
        )
        # scaling conv
        x = self.conv2(x)
        return x


class ChannelAttn(nn.Module):
    """Channel Attention (Sec. 3.1.I.2)"""

    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttn, self).__init__()
        assert in_channels % reduction_rate == 0
        self.conv1 = ConvBlock(in_channels, in_channels // reduction_rate, 1)
        self.conv2 = ConvBlock(in_channels // reduction_rate, in_channels, 1)

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:])
        # excitation operation (2 conv layers)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SoftAttn(nn.Module):
    """Soft Attention (Sec. 3.1.I)
    
    Aim: Spatial Attention + Channel Attention
    
    Output: attention maps with shape identical to input.
    """

    def __init__(self, in_channels):
        super(SoftAttn, self).__init__()
        self.spatial_attn = SpatialAttn()
        self.channel_attn = ChannelAttn(in_channels)
        self.conv = ConvBlock(in_channels, in_channels, 1)

    def forward(self, x):
        y_spatial = self.spatial_attn(x)
        y_channel = self.channel_attn(x)
        y = y_spatial * y_channel
        y = torch.sigmoid(self.conv(y))
        return y


class HardAttn(nn.Module):
    """Hard Attention (Sec. 3.1.II)"""

    def __init__(self, in_channels):
        super(HardAttn, self).__init__()
        self.fc = nn.Linear(in_channels, 4 * 2)
        self.init_params()

    def init_params(self):
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(
            torch.tensor(
                [0, -0.75, 0, -0.25, 0, 0.25, 0, 0.75], dtype=torch.float
            )
        )

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), x.size(1))
        # predict transformation parameters
        theta = torch.tanh(self.fc(x))
        theta = theta.view(-1, 4, 2)
        return theta


class HarmAttn(nn.Module):
    """Harmonious Attention (Sec. 3.1)"""

    def __init__(self, in_channels):
        super(HarmAttn, self).__init__()
        self.soft_attn = SoftAttn(in_channels)
        self.hard_attn = HardAttn(in_channels)

    def forward(self, x):
        y_soft_attn = self.soft_attn(x)
        theta = self.hard_attn(x)
        return y_soft_attn, theta


class HACNN(nn.Module):
    """Harmonious Attention Convolutional Neural Network.

    Reference:
        Li et al. Harmonious Attention Network for Person Re-identification. CVPR 2018.

    Public keys:
        - ``hacnn``: HACNN.
    """

    # Args:
    #    num_classes (int): number of classes to predict
    #    nchannels (list): number of channels AFTER concatenation
    #    feat_dim (int): feature dimension for a single stream
    #    learn_region (bool): whether to learn region features (i.e. local branch)

    def __init__(
        self,
        num_classes,
        loss='softmax',
        nchannels=[128, 256, 384],
        feat_dim=512,
        learn_region=True,
        use_gpu=True,
        **kwargs
    ):
        super(HACNN, self).__init__()
        self.loss = loss
        self.learn_region = learn_region
        self.use_gpu = use_gpu

        self.conv = ConvBlock(3, 32, 3, s=2, p=1)

        # Construct Inception + HarmAttn blocks
        # ============== Block 1 ==============
        self.inception1 = nn.Sequential(
            InceptionA(32, nchannels[0]),
            InceptionB(nchannels[0], nchannels[0]),
        )
        self.ha1 = HarmAttn(nchannels[0])

        # ============== Block 2 ==============
        self.inception2 = nn.Sequential(
            InceptionA(nchannels[0], nchannels[1]),
            InceptionB(nchannels[1], nchannels[1]),
        )
        self.ha2 = HarmAttn(nchannels[1])

        # ============== Block 3 ==============
        self.inception3 = nn.Sequential(
            InceptionA(nchannels[1], nchannels[2]),
            InceptionB(nchannels[2], nchannels[2]),
        )
        self.ha3 = HarmAttn(nchannels[2])

        self.fc_global = nn.Sequential(
            nn.Linear(nchannels[2], feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
        )
        self.classifier_global = nn.Linear(feat_dim, num_classes)

        if self.learn_region:
            self.init_scale_factors()
            self.local_conv1 = InceptionB(32, nchannels[0])
            self.local_conv2 = InceptionB(nchannels[0], nchannels[1])
            self.local_conv3 = InceptionB(nchannels[1], nchannels[2])
            self.fc_local = nn.Sequential(
                nn.Linear(nchannels[2] * 4, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU(),
            )
            self.classifier_local = nn.Linear(feat_dim, num_classes)
            self.feat_dim = feat_dim * 2
        else:
            self.feat_dim = feat_dim

    def init_scale_factors(self):
        # initialize scale factors (s_w, s_h) for four regions
        self.scale_factors = []
        self.scale_factors.append(
            torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float)
        )
        self.scale_factors.append(
            torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float)
        )
        self.scale_factors.append(
            torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float)
        )
        self.scale_factors.append(
            torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float)
        )

    def stn(self, x, theta):
        """Performs spatial transform
        
        x: (batch, channel, height, width)
        theta: (batch, 2, 3)
        """
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def transform_theta(self, theta_i, region_idx):
        """Transforms theta to include (s_w, s_h), resulting in (batch, 2, 3)"""
        scale_factors = self.scale_factors[region_idx]
        theta = torch.zeros(theta_i.size(0), 2, 3)
        theta[:, :, :2] = scale_factors
        theta[:, :, -1] = theta_i
        if self.use_gpu:
            theta = theta.cuda()
        return theta

    def forward(self, x):
        assert x.size(2) == 160 and x.size(3) == 64, \
            'Input size does not match, expected (160, 64) but got ({}, {})'.format(x.size(2), x.size(3))
        x = self.conv(x)

        # ============== Block 1 ==============
        # global branch
        x1 = self.inception1(x)
        x1_attn, x1_theta = self.ha1(x1)
        x1_out = x1 * x1_attn
        # local branch
        if self.learn_region:
            x1_local_list = []
            for region_idx in range(4):
                x1_theta_i = x1_theta[:, region_idx, :]
                x1_theta_i = self.transform_theta(x1_theta_i, region_idx)
                x1_trans_i = self.stn(x, x1_theta_i)
                x1_trans_i = F.upsample(
                    x1_trans_i, (24, 28), mode='bilinear', align_corners=True
                )
                x1_local_i = self.local_conv1(x1_trans_i)
                x1_local_list.append(x1_local_i)

        # ============== Block 2 ==============
        # Block 2
        # global branch
        x2 = self.inception2(x1_out)
        x2_attn, x2_theta = self.ha2(x2)
        x2_out = x2 * x2_attn
        # local branch
        if self.learn_region:
            x2_local_list = []
            for region_idx in range(4):
                x2_theta_i = x2_theta[:, region_idx, :]
                x2_theta_i = self.transform_theta(x2_theta_i, region_idx)
                x2_trans_i = self.stn(x1_out, x2_theta_i)
                x2_trans_i = F.upsample(
                    x2_trans_i, (12, 14), mode='bilinear', align_corners=True
                )
                x2_local_i = x2_trans_i + x1_local_list[region_idx]
                x2_local_i = self.local_conv2(x2_local_i)
                x2_local_list.append(x2_local_i)

        # ============== Block 3 ==============
        # Block 3
        # global branch
        x3 = self.inception3(x2_out)
        x3_attn, x3_theta = self.ha3(x3)
        x3_out = x3 * x3_attn
        # local branch
        if self.learn_region:
            x3_local_list = []
            for region_idx in range(4):
                x3_theta_i = x3_theta[:, region_idx, :]
                x3_theta_i = self.transform_theta(x3_theta_i, region_idx)
                x3_trans_i = self.stn(x2_out, x3_theta_i)
                x3_trans_i = F.upsample(
                    x3_trans_i, (6, 7), mode='bilinear', align_corners=True
                )
                x3_local_i = x3_trans_i + x2_local_list[region_idx]
                x3_local_i = self.local_conv3(x3_local_i)
                x3_local_list.append(x3_local_i)

        # ============== Feature generation ==============
        # global branch
        x_global = F.avg_pool2d(x3_out,
                                x3_out.size()[2:]
                                ).view(x3_out.size(0), x3_out.size(1))
        x_global = self.fc_global(x_global)
        # local branch
        if self.learn_region:
            x_local_list = []
            for region_idx in range(4):
                x_local_i = x3_local_list[region_idx]
                x_local_i = F.avg_pool2d(x_local_i,
                                         x_local_i.size()[2:]
                                         ).view(x_local_i.size(0), -1)
                x_local_list.append(x_local_i)
            x_local = torch.cat(x_local_list, 1)
            x_local = self.fc_local(x_local)

        if not self.training:
            # l2 normalization before concatenation
            if self.learn_region:
                x_global = x_global / x_global.norm(p=2, dim=1, keepdim=True)
                x_local = x_local / x_local.norm(p=2, dim=1, keepdim=True)
                return torch.cat([x_global, x_local], 1)
            else:
                return x_global

        prelogits_global = self.classifier_global(x_global)
        if self.learn_region:
            prelogits_local = self.classifier_local(x_local)

        if self.loss == 'softmax':
            if self.learn_region:
                return (prelogits_global, prelogits_local)
            else:
                return prelogits_global

        elif self.loss == 'triplet':
            if self.learn_region:
                return (prelogits_global, prelogits_local), (x_global, x_local)
            else:
                return prelogits_global, x_global

        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
