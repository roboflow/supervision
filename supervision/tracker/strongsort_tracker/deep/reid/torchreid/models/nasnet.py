from __future__ import division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['nasnetamobile']
"""
NASNet Mobile
Thanks to Anastasiia (https://github.com/DagnyT) for the great help, support and motivation!


------------------------------------------------------------------------------------
      Architecture       | Top-1 Acc | Top-5 Acc |  Multiply-Adds |  Params (M)
------------------------------------------------------------------------------------
|   NASNet-A (4 @ 1056)  |   74.08%  |   91.74%  |       564 M    |     5.3        |
------------------------------------------------------------------------------------
# References:
 - [Learning Transferable Architectures for Scalable Image Recognition]
    (https://arxiv.org/abs/1707.07012)
"""
"""
Code imported from https://github.com/Cadene/pretrained-models.pytorch
"""

pretrained_settings = {
    'nasnetamobile': {
        'imagenet': {
            # 'url': 'https://github.com/veronikayurchuk/pretrained-models.pytorch/releases/download/v1.0/nasnetmobile-7e03cead.pth.tar',
            'url':
            'http://data.lip6.fr/cadene/pretrainedmodels/nasnetamobile-7e03cead.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224], # resize 256
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000
        },
        # 'imagenet+background': {
        #     # 'url': 'http://data.lip6.fr/cadene/pretrainedmodels/nasnetalarge-a1897284.pth',
        #     'input_space': 'RGB',
        #     'input_size': [3, 224, 224], # resize 256
        #     'input_range': [0, 1],
        #     'mean': [0.5, 0.5, 0.5],
        #     'std': [0.5, 0.5, 0.5],
        #     'num_classes': 1001
        # }
    }
}


class MaxPoolPad(nn.Module):

    def __init__(self):
        super(MaxPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:].contiguous()
        return x


class AvgPoolPad(nn.Module):

    def __init__(self, stride=2, padding=1):
        super(AvgPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.AvgPool2d(
            3, stride=stride, padding=padding, count_include_pad=False
        )

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:].contiguous()
        return x


class SeparableConv2d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        dw_kernel,
        dw_stride,
        dw_padding,
        bias=False
    ):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(
            in_channels,
            in_channels,
            dw_kernel,
            stride=dw_stride,
            padding=dw_padding,
            bias=bias,
            groups=in_channels
        )
        self.pointwise_conv2d = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, bias=bias
        )

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        name=None,
        bias=False
    ):
        super(BranchSeparables, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(
            in_channels, in_channels, kernel_size, stride, padding, bias=bias
        )
        self.bn_sep_1 = nn.BatchNorm2d(
            in_channels, eps=0.001, momentum=0.1, affine=True
        )
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(
            in_channels, out_channels, kernel_size, 1, padding, bias=bias
        )
        self.bn_sep_2 = nn.BatchNorm2d(
            out_channels, eps=0.001, momentum=0.1, affine=True
        )
        self.name = name

    def forward(self, x):
        x = self.relu(x)
        if self.name == 'specific':
            x = nn.ZeroPad2d((1, 0, 1, 0))(x)
        x = self.separable_1(x)
        if self.name == 'specific':
            x = x[:, :, 1:, 1:].contiguous()

        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class BranchSeparablesStem(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias=False
    ):
        super(BranchSeparablesStem, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.bn_sep_1 = nn.BatchNorm2d(
            out_channels, eps=0.001, momentum=0.1, affine=True
        )
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(
            out_channels, out_channels, kernel_size, 1, padding, bias=bias
        )
        self.bn_sep_2 = nn.BatchNorm2d(
            out_channels, eps=0.001, momentum=0.1, affine=True
        )

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class BranchSeparablesReduction(BranchSeparables):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        z_padding=1,
        bias=False
    ):
        BranchSeparables.__init__(
            self, in_channels, out_channels, kernel_size, stride, padding, bias
        )
        self.padding = nn.ZeroPad2d((z_padding, 0, z_padding, 0))

    def forward(self, x):
        x = self.relu(x)
        x = self.padding(x)
        x = self.separable_1(x)
        x = x[:, :, 1:, 1:].contiguous()
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class CellStem0(nn.Module):

    def __init__(self, stem_filters, num_filters=42):
        super(CellStem0, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module(
            'conv',
            nn.Conv2d(
                self.stem_filters, self.num_filters, 1, stride=1, bias=False
            )
        )
        self.conv_1x1.add_module(
            'bn',
            nn.BatchNorm2d(
                self.num_filters, eps=0.001, momentum=0.1, affine=True
            )
        )

        self.comb_iter_0_left = BranchSeparables(
            self.num_filters, self.num_filters, 5, 2, 2
        )
        self.comb_iter_0_right = BranchSeparablesStem(
            self.stem_filters, self.num_filters, 7, 2, 3, bias=False
        )

        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparablesStem(
            self.stem_filters, self.num_filters, 7, 2, 3, bias=False
        )

        self.comb_iter_2_left = nn.AvgPool2d(
            3, stride=2, padding=1, count_include_pad=False
        )
        self.comb_iter_2_right = BranchSeparablesStem(
            self.stem_filters, self.num_filters, 5, 2, 2, bias=False
        )

        self.comb_iter_3_right = nn.AvgPool2d(
            3, stride=1, padding=1, count_include_pad=False
        )

        self.comb_iter_4_left = BranchSeparables(
            self.num_filters, self.num_filters, 3, 1, 1, bias=False
        )
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x1)
        x_comb_iter_0_right = self.comb_iter_0_right(x)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x1)
        x_comb_iter_1_right = self.comb_iter_1_right(x)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x1)
        x_comb_iter_2_right = self.comb_iter_2_right(x)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x1)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat(
            [x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1
        )
        return x_out


class CellStem1(nn.Module):

    def __init__(self, stem_filters, num_filters):
        super(CellStem1, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module(
            'conv',
            nn.Conv2d(
                2 * self.num_filters,
                self.num_filters,
                1,
                stride=1,
                bias=False
            )
        )
        self.conv_1x1.add_module(
            'bn',
            nn.BatchNorm2d(
                self.num_filters, eps=0.001, momentum=0.1, affine=True
            )
        )

        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module(
            'avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False)
        )
        self.path_1.add_module(
            'conv',
            nn.Conv2d(
                self.stem_filters,
                self.num_filters // 2,
                1,
                stride=1,
                bias=False
            )
        )
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module(
            'avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False)
        )
        self.path_2.add_module(
            'conv',
            nn.Conv2d(
                self.stem_filters,
                self.num_filters // 2,
                1,
                stride=1,
                bias=False
            )
        )

        self.final_path_bn = nn.BatchNorm2d(
            self.num_filters, eps=0.001, momentum=0.1, affine=True
        )

        self.comb_iter_0_left = BranchSeparables(
            self.num_filters,
            self.num_filters,
            5,
            2,
            2,
            name='specific',
            bias=False
        )
        self.comb_iter_0_right = BranchSeparables(
            self.num_filters,
            self.num_filters,
            7,
            2,
            3,
            name='specific',
            bias=False
        )

        # self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparables(
            self.num_filters,
            self.num_filters,
            7,
            2,
            3,
            name='specific',
            bias=False
        )

        # self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparables(
            self.num_filters,
            self.num_filters,
            5,
            2,
            2,
            name='specific',
            bias=False
        )

        self.comb_iter_3_right = nn.AvgPool2d(
            3, stride=1, padding=1, count_include_pad=False
        )

        self.comb_iter_4_left = BranchSeparables(
            self.num_filters,
            self.num_filters,
            3,
            1,
            1,
            name='specific',
            bias=False
        )
        # self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x_conv0, x_stem_0):
        x_left = self.conv_1x1(x_stem_0)

        x_relu = self.relu(x_conv0)
        # path 1
        x_path1 = self.path_1(x_relu)
        # path 2
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        # final path
        x_right = self.final_path_bn(torch.cat([x_path1, x_path2], 1))

        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_right)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_left)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_left)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat(
            [x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1
        )
        return x_out


class FirstCell(nn.Module):

    def __init__(
        self, in_channels_left, out_channels_left, in_channels_right,
        out_channels_right
    ):
        super(FirstCell, self).__init__()
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module(
            'conv',
            nn.Conv2d(
                in_channels_right, out_channels_right, 1, stride=1, bias=False
            )
        )
        self.conv_1x1.add_module(
            'bn',
            nn.BatchNorm2d(
                out_channels_right, eps=0.001, momentum=0.1, affine=True
            )
        )

        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module(
            'avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False)
        )
        self.path_1.add_module(
            'conv',
            nn.Conv2d(
                in_channels_left, out_channels_left, 1, stride=1, bias=False
            )
        )
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module(
            'avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False)
        )
        self.path_2.add_module(
            'conv',
            nn.Conv2d(
                in_channels_left, out_channels_left, 1, stride=1, bias=False
            )
        )

        self.final_path_bn = nn.BatchNorm2d(
            out_channels_left * 2, eps=0.001, momentum=0.1, affine=True
        )

        self.comb_iter_0_left = BranchSeparables(
            out_channels_right, out_channels_right, 5, 1, 2, bias=False
        )
        self.comb_iter_0_right = BranchSeparables(
            out_channels_right, out_channels_right, 3, 1, 1, bias=False
        )

        self.comb_iter_1_left = BranchSeparables(
            out_channels_right, out_channels_right, 5, 1, 2, bias=False
        )
        self.comb_iter_1_right = BranchSeparables(
            out_channels_right, out_channels_right, 3, 1, 1, bias=False
        )

        self.comb_iter_2_left = nn.AvgPool2d(
            3, stride=1, padding=1, count_include_pad=False
        )

        self.comb_iter_3_left = nn.AvgPool2d(
            3, stride=1, padding=1, count_include_pad=False
        )
        self.comb_iter_3_right = nn.AvgPool2d(
            3, stride=1, padding=1, count_include_pad=False
        )

        self.comb_iter_4_left = BranchSeparables(
            out_channels_right, out_channels_right, 3, 1, 1, bias=False
        )

    def forward(self, x, x_prev):
        x_relu = self.relu(x_prev)
        # path 1
        x_path1 = self.path_1(x_relu)
        # path 2
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        # final path
        x_left = self.final_path_bn(torch.cat([x_path1, x_path2], 1))

        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left

        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right

        x_out = torch.cat(
            [
                x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2,
                x_comb_iter_3, x_comb_iter_4
            ], 1
        )
        return x_out


class NormalCell(nn.Module):

    def __init__(
        self, in_channels_left, out_channels_left, in_channels_right,
        out_channels_right
    ):
        super(NormalCell, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module(
            'conv',
            nn.Conv2d(
                in_channels_left, out_channels_left, 1, stride=1, bias=False
            )
        )
        self.conv_prev_1x1.add_module(
            'bn',
            nn.BatchNorm2d(
                out_channels_left, eps=0.001, momentum=0.1, affine=True
            )
        )

        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module(
            'conv',
            nn.Conv2d(
                in_channels_right, out_channels_right, 1, stride=1, bias=False
            )
        )
        self.conv_1x1.add_module(
            'bn',
            nn.BatchNorm2d(
                out_channels_right, eps=0.001, momentum=0.1, affine=True
            )
        )

        self.comb_iter_0_left = BranchSeparables(
            out_channels_right, out_channels_right, 5, 1, 2, bias=False
        )
        self.comb_iter_0_right = BranchSeparables(
            out_channels_left, out_channels_left, 3, 1, 1, bias=False
        )

        self.comb_iter_1_left = BranchSeparables(
            out_channels_left, out_channels_left, 5, 1, 2, bias=False
        )
        self.comb_iter_1_right = BranchSeparables(
            out_channels_left, out_channels_left, 3, 1, 1, bias=False
        )

        self.comb_iter_2_left = nn.AvgPool2d(
            3, stride=1, padding=1, count_include_pad=False
        )

        self.comb_iter_3_left = nn.AvgPool2d(
            3, stride=1, padding=1, count_include_pad=False
        )
        self.comb_iter_3_right = nn.AvgPool2d(
            3, stride=1, padding=1, count_include_pad=False
        )

        self.comb_iter_4_left = BranchSeparables(
            out_channels_right, out_channels_right, 3, 1, 1, bias=False
        )

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left

        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right

        x_out = torch.cat(
            [
                x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2,
                x_comb_iter_3, x_comb_iter_4
            ], 1
        )
        return x_out


class ReductionCell0(nn.Module):

    def __init__(
        self, in_channels_left, out_channels_left, in_channels_right,
        out_channels_right
    ):
        super(ReductionCell0, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module(
            'conv',
            nn.Conv2d(
                in_channels_left, out_channels_left, 1, stride=1, bias=False
            )
        )
        self.conv_prev_1x1.add_module(
            'bn',
            nn.BatchNorm2d(
                out_channels_left, eps=0.001, momentum=0.1, affine=True
            )
        )

        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module(
            'conv',
            nn.Conv2d(
                in_channels_right, out_channels_right, 1, stride=1, bias=False
            )
        )
        self.conv_1x1.add_module(
            'bn',
            nn.BatchNorm2d(
                out_channels_right, eps=0.001, momentum=0.1, affine=True
            )
        )

        self.comb_iter_0_left = BranchSeparablesReduction(
            out_channels_right, out_channels_right, 5, 2, 2, bias=False
        )
        self.comb_iter_0_right = BranchSeparablesReduction(
            out_channels_right, out_channels_right, 7, 2, 3, bias=False
        )

        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparablesReduction(
            out_channels_right, out_channels_right, 7, 2, 3, bias=False
        )

        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparablesReduction(
            out_channels_right, out_channels_right, 5, 2, 2, bias=False
        )

        self.comb_iter_3_right = nn.AvgPool2d(
            3, stride=1, padding=1, count_include_pad=False
        )

        self.comb_iter_4_left = BranchSeparablesReduction(
            out_channels_right, out_channels_right, 3, 1, 1, bias=False
        )
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat(
            [x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1
        )
        return x_out


class ReductionCell1(nn.Module):

    def __init__(
        self, in_channels_left, out_channels_left, in_channels_right,
        out_channels_right
    ):
        super(ReductionCell1, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module(
            'conv',
            nn.Conv2d(
                in_channels_left, out_channels_left, 1, stride=1, bias=False
            )
        )
        self.conv_prev_1x1.add_module(
            'bn',
            nn.BatchNorm2d(
                out_channels_left, eps=0.001, momentum=0.1, affine=True
            )
        )

        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module(
            'conv',
            nn.Conv2d(
                in_channels_right, out_channels_right, 1, stride=1, bias=False
            )
        )
        self.conv_1x1.add_module(
            'bn',
            nn.BatchNorm2d(
                out_channels_right, eps=0.001, momentum=0.1, affine=True
            )
        )

        self.comb_iter_0_left = BranchSeparables(
            out_channels_right,
            out_channels_right,
            5,
            2,
            2,
            name='specific',
            bias=False
        )
        self.comb_iter_0_right = BranchSeparables(
            out_channels_right,
            out_channels_right,
            7,
            2,
            3,
            name='specific',
            bias=False
        )

        # self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparables(
            out_channels_right,
            out_channels_right,
            7,
            2,
            3,
            name='specific',
            bias=False
        )

        # self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparables(
            out_channels_right,
            out_channels_right,
            5,
            2,
            2,
            name='specific',
            bias=False
        )

        self.comb_iter_3_right = nn.AvgPool2d(
            3, stride=1, padding=1, count_include_pad=False
        )

        self.comb_iter_4_left = BranchSeparables(
            out_channels_right,
            out_channels_right,
            3,
            1,
            1,
            name='specific',
            bias=False
        )
        # self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat(
            [x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1
        )
        return x_out


class NASNetAMobile(nn.Module):
    """Neural Architecture Search (NAS).

    Reference:
        Zoph et al. Learning Transferable Architectures
        for Scalable Image Recognition. CVPR 2018.

    Public keys:
        - ``nasnetamobile``: NASNet-A Mobile.
    """

    def __init__(
        self,
        num_classes,
        loss,
        stem_filters=32,
        penultimate_filters=1056,
        filters_multiplier=2,
        **kwargs
    ):
        super(NASNetAMobile, self).__init__()
        self.stem_filters = stem_filters
        self.penultimate_filters = penultimate_filters
        self.filters_multiplier = filters_multiplier
        self.loss = loss

        filters = self.penultimate_filters // 24
        # 24 is default value for the architecture

        self.conv0 = nn.Sequential()
        self.conv0.add_module(
            'conv',
            nn.Conv2d(
                in_channels=3,
                out_channels=self.stem_filters,
                kernel_size=3,
                padding=0,
                stride=2,
                bias=False
            )
        )
        self.conv0.add_module(
            'bn',
            nn.BatchNorm2d(
                self.stem_filters, eps=0.001, momentum=0.1, affine=True
            )
        )

        self.cell_stem_0 = CellStem0(
            self.stem_filters, num_filters=filters // (filters_multiplier**2)
        )
        self.cell_stem_1 = CellStem1(
            self.stem_filters, num_filters=filters // filters_multiplier
        )

        self.cell_0 = FirstCell(
            in_channels_left=filters,
            out_channels_left=filters // 2, # 1, 0.5
            in_channels_right=2 * filters,
            out_channels_right=filters
        ) # 2, 1
        self.cell_1 = NormalCell(
            in_channels_left=2 * filters,
            out_channels_left=filters, # 2, 1
            in_channels_right=6 * filters,
            out_channels_right=filters
        ) # 6, 1
        self.cell_2 = NormalCell(
            in_channels_left=6 * filters,
            out_channels_left=filters, # 6, 1
            in_channels_right=6 * filters,
            out_channels_right=filters
        ) # 6, 1
        self.cell_3 = NormalCell(
            in_channels_left=6 * filters,
            out_channels_left=filters, # 6, 1
            in_channels_right=6 * filters,
            out_channels_right=filters
        ) # 6, 1

        self.reduction_cell_0 = ReductionCell0(
            in_channels_left=6 * filters,
            out_channels_left=2 * filters, # 6, 2
            in_channels_right=6 * filters,
            out_channels_right=2 * filters
        ) # 6, 2

        self.cell_6 = FirstCell(
            in_channels_left=6 * filters,
            out_channels_left=filters, # 6, 1
            in_channels_right=8 * filters,
            out_channels_right=2 * filters
        ) # 8, 2
        self.cell_7 = NormalCell(
            in_channels_left=8 * filters,
            out_channels_left=2 * filters, # 8, 2
            in_channels_right=12 * filters,
            out_channels_right=2 * filters
        ) # 12, 2
        self.cell_8 = NormalCell(
            in_channels_left=12 * filters,
            out_channels_left=2 * filters, # 12, 2
            in_channels_right=12 * filters,
            out_channels_right=2 * filters
        ) # 12, 2
        self.cell_9 = NormalCell(
            in_channels_left=12 * filters,
            out_channels_left=2 * filters, # 12, 2
            in_channels_right=12 * filters,
            out_channels_right=2 * filters
        ) # 12, 2

        self.reduction_cell_1 = ReductionCell1(
            in_channels_left=12 * filters,
            out_channels_left=4 * filters, # 12, 4
            in_channels_right=12 * filters,
            out_channels_right=4 * filters
        ) # 12, 4

        self.cell_12 = FirstCell(
            in_channels_left=12 * filters,
            out_channels_left=2 * filters, # 12, 2
            in_channels_right=16 * filters,
            out_channels_right=4 * filters
        ) # 16, 4
        self.cell_13 = NormalCell(
            in_channels_left=16 * filters,
            out_channels_left=4 * filters, # 16, 4
            in_channels_right=24 * filters,
            out_channels_right=4 * filters
        ) # 24, 4
        self.cell_14 = NormalCell(
            in_channels_left=24 * filters,
            out_channels_left=4 * filters, # 24, 4
            in_channels_right=24 * filters,
            out_channels_right=4 * filters
        ) # 24, 4
        self.cell_15 = NormalCell(
            in_channels_left=24 * filters,
            out_channels_left=4 * filters, # 24, 4
            in_channels_right=24 * filters,
            out_channels_right=4 * filters
        ) # 24, 4

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(24 * filters, num_classes)

        self._init_params()

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

    def features(self, input):
        x_conv0 = self.conv0(input)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)

        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)

        x_reduction_cell_0 = self.reduction_cell_0(x_cell_3, x_cell_2)

        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_3)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)

        x_reduction_cell_1 = self.reduction_cell_1(x_cell_9, x_cell_8)

        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_9)
        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)

        x_cell_15 = self.relu(x_cell_15)
        x_cell_15 = F.avg_pool2d(
            x_cell_15,
            x_cell_15.size()[2:]
        ) # global average pool
        x_cell_15 = x_cell_15.view(x_cell_15.size(0), -1)
        x_cell_15 = self.dropout(x_cell_15)

        return x_cell_15

    def forward(self, input):
        v = self.features(input)

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
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


def nasnetamobile(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = NASNetAMobile(num_classes, loss, **kwargs)
    if pretrained:
        model_url = pretrained_settings['nasnetamobile']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model
