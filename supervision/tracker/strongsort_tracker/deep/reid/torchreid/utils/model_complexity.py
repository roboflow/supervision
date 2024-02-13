from __future__ import absolute_import, division, print_function

import math
from collections import defaultdict, namedtuple
from itertools import repeat

import numpy as np
import torch

__all__ = ["compute_model_complexity"]
"""
Utility
"""


def _ntuple(n):
    def parse(x):
        if isinstance(x, int):
            return tuple(repeat(x, n))
        return x

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
"""
Convolution
"""


def hook_convNd(m, x, y):
    k = torch.prod(torch.Tensor(m.kernel_size)).item()
    cin = m.in_channels
    flops_per_ele = k * cin  # + (k*cin-1)
    if m.bias is not None:
        flops_per_ele += 1
    flops = flops_per_ele * y.numel() / m.groups
    return int(flops)


"""
Pooling
"""


def hook_maxpool1d(m, x, y):
    flops_per_ele = m.kernel_size - 1
    flops = flops_per_ele * y.numel()
    return int(flops)


def hook_maxpool2d(m, x, y):
    k = _pair(m.kernel_size)
    k = torch.prod(torch.Tensor(k)).item()
    # ops: compare
    flops_per_ele = k - 1
    flops = flops_per_ele * y.numel()
    return int(flops)


def hook_maxpool3d(m, x, y):
    k = _triple(m.kernel_size)
    k = torch.prod(torch.Tensor(k)).item()
    flops_per_ele = k - 1
    flops = flops_per_ele * y.numel()
    return int(flops)


def hook_avgpool1d(m, x, y):
    flops_per_ele = m.kernel_size
    flops = flops_per_ele * y.numel()
    return int(flops)


def hook_avgpool2d(m, x, y):
    k = _pair(m.kernel_size)
    k = torch.prod(torch.Tensor(k)).item()
    flops_per_ele = k
    flops = flops_per_ele * y.numel()
    return int(flops)


def hook_avgpool3d(m, x, y):
    k = _triple(m.kernel_size)
    k = torch.prod(torch.Tensor(k)).item()
    flops_per_ele = k
    flops = flops_per_ele * y.numel()
    return int(flops)


def hook_adapmaxpool1d(m, x, y):
    x = x[0]
    out_size = m.output_size
    k = math.ceil(x.size(2) / out_size)
    flops_per_ele = k - 1
    flops = flops_per_ele * y.numel()
    return int(flops)


def hook_adapmaxpool2d(m, x, y):
    x = x[0]
    out_size = _pair(m.output_size)
    k = torch.Tensor(list(x.size()[2:])) / torch.Tensor(out_size)
    k = torch.prod(torch.ceil(k)).item()
    flops_per_ele = k - 1
    flops = flops_per_ele * y.numel()
    return int(flops)


def hook_adapmaxpool3d(m, x, y):
    x = x[0]
    out_size = _triple(m.output_size)
    k = torch.Tensor(list(x.size()[2:])) / torch.Tensor(out_size)
    k = torch.prod(torch.ceil(k)).item()
    flops_per_ele = k - 1
    flops = flops_per_ele * y.numel()
    return int(flops)


def hook_adapavgpool1d(m, x, y):
    x = x[0]
    out_size = m.output_size
    k = math.ceil(x.size(2) / out_size)
    flops_per_ele = k
    flops = flops_per_ele * y.numel()
    return int(flops)


def hook_adapavgpool2d(m, x, y):
    x = x[0]
    out_size = _pair(m.output_size)
    k = torch.Tensor(list(x.size()[2:])) / torch.Tensor(out_size)
    k = torch.prod(torch.ceil(k)).item()
    flops_per_ele = k
    flops = flops_per_ele * y.numel()
    return int(flops)


def hook_adapavgpool3d(m, x, y):
    x = x[0]
    out_size = _triple(m.output_size)
    k = torch.Tensor(list(x.size()[2:])) / torch.Tensor(out_size)
    k = torch.prod(torch.ceil(k)).item()
    flops_per_ele = k
    flops = flops_per_ele * y.numel()
    return int(flops)


"""
Non-linear activations
"""


def hook_relu(m, x, y):
    # eq: max(0, x)
    num_ele = y.numel()
    return int(num_ele)


def hook_leakyrelu(m, x, y):
    # eq: max(0, x) + negative_slope*min(0, x)
    num_ele = y.numel()
    flops = 3 * num_ele
    return int(flops)


"""
Normalization
"""


def hook_batchnormNd(m, x, y):
    num_ele = y.numel()
    flops = 2 * num_ele  # mean and std
    if m.affine:
        flops += 2 * num_ele  # gamma and beta
    return int(flops)


def hook_instancenormNd(m, x, y):
    return hook_batchnormNd(m, x, y)


def hook_groupnorm(m, x, y):
    return hook_batchnormNd(m, x, y)


def hook_layernorm(m, x, y):
    num_ele = y.numel()
    flops = 2 * num_ele  # mean and std
    if m.elementwise_affine:
        flops += 2 * num_ele  # gamma and beta
    return int(flops)


"""
Linear
"""


def hook_linear(m, x, y):
    flops_per_ele = m.in_features  # + (m.in_features-1)
    if m.bias is not None:
        flops_per_ele += 1
    flops = flops_per_ele * y.numel()
    return int(flops)


__generic_flops_counter = {
    # Convolution
    "Conv1d": hook_convNd,
    "Conv2d": hook_convNd,
    "Conv3d": hook_convNd,
    # Pooling
    "MaxPool1d": hook_maxpool1d,
    "MaxPool2d": hook_maxpool2d,
    "MaxPool3d": hook_maxpool3d,
    "AvgPool1d": hook_avgpool1d,
    "AvgPool2d": hook_avgpool2d,
    "AvgPool3d": hook_avgpool3d,
    "AdaptiveMaxPool1d": hook_adapmaxpool1d,
    "AdaptiveMaxPool2d": hook_adapmaxpool2d,
    "AdaptiveMaxPool3d": hook_adapmaxpool3d,
    "AdaptiveAvgPool1d": hook_adapavgpool1d,
    "AdaptiveAvgPool2d": hook_adapavgpool2d,
    "AdaptiveAvgPool3d": hook_adapavgpool3d,
    # Non-linear activations
    "ReLU": hook_relu,
    "ReLU6": hook_relu,
    "LeakyReLU": hook_leakyrelu,
    # Normalization
    "BatchNorm1d": hook_batchnormNd,
    "BatchNorm2d": hook_batchnormNd,
    "BatchNorm3d": hook_batchnormNd,
    "InstanceNorm1d": hook_instancenormNd,
    "InstanceNorm2d": hook_instancenormNd,
    "InstanceNorm3d": hook_instancenormNd,
    "GroupNorm": hook_groupnorm,
    "LayerNorm": hook_layernorm,
    # Linear
    "Linear": hook_linear,
}

__conv_linear_flops_counter = {
    # Convolution
    "Conv1d": hook_convNd,
    "Conv2d": hook_convNd,
    "Conv3d": hook_convNd,
    # Linear
    "Linear": hook_linear,
}


def _get_flops_counter(only_conv_linear):
    if only_conv_linear:
        return __conv_linear_flops_counter
    return __generic_flops_counter


def compute_model_complexity(model, input_size, verbose=False, only_conv_linear=True):
    """Returns number of parameters and FLOPs.

    .. note::
        (1) this function only provides an estimate of the theoretical time complexity
        rather than the actual running time which depends on implementations and hardware,
        and (2) the FLOPs is only counted for layers that are used at test time. This means
        that redundant layers such as person ID classification layer will be ignored as it
        is discarded when doing feature extraction. Note that the inference graph depends on
        how you construct the computations in ``forward()``.

    Args:
        model (nn.Module): network model.
        input_size (tuple): input size, e.g. (1, 3, 256, 128).
        verbose (bool, optional): shows detailed complexity of
            each module. Default is False.
        only_conv_linear (bool, optional): only considers convolution
            and linear layers when counting flops. Default is True.
            If set to False, flops of all layers will be counted.

    Examples::
        >>> from supervision.tracker.stronsort_tracker.torchreid import models, utils
        >>> model = models.build_model(name='resnet50', num_classes=1000)
        >>> num_params, flops = utils.compute_model_complexity(model, (1, 3, 256, 128), verbose=True)
    """
    registered_handles = []
    layer_list = []
    layer = namedtuple("layer", ["class_name", "params", "flops"])

    def _add_hooks(m):
        def _has_submodule(m):
            return len(list(m.children())) > 0

        def _hook(m, x, y):
            params = sum(p.numel() for p in m.parameters())
            class_name = str(m.__class__.__name__)
            flops_counter = _get_flops_counter(only_conv_linear)
            if class_name in flops_counter:
                flops = flops_counter[class_name](m, x, y)
            else:
                flops = 0
            layer_list.append(layer(class_name=class_name, params=params, flops=flops))

        # only consider the very basic nn layer
        if _has_submodule(m):
            return

        handle = m.register_forward_hook(_hook)
        registered_handles.append(handle)

    default_train_mode = model.training

    model.eval().apply(_add_hooks)
    input = torch.rand(input_size)
    if next(model.parameters()).is_cuda:
        input = input.cuda()
    model(input)  # forward

    for handle in registered_handles:
        handle.remove()

    model.train(default_train_mode)

    if verbose:
        per_module_params = defaultdict(list)
        per_module_flops = defaultdict(list)

    total_params, total_flops = 0, 0

    for layer in layer_list:
        total_params += layer.params
        total_flops += layer.flops
        if verbose:
            per_module_params[layer.class_name].append(layer.params)
            per_module_flops[layer.class_name].append(layer.flops)

    if verbose:
        num_udscore = 55
        print("  {}".format("-" * num_udscore))
        print("  Model complexity with input size {}".format(input_size))
        print("  {}".format("-" * num_udscore))
        for class_name in per_module_params:
            params = int(np.sum(per_module_params[class_name]))
            flops = int(np.sum(per_module_flops[class_name]))
            print("  {} (params={:,}, flops={:,})".format(class_name, params, flops))
        print("  {}".format("-" * num_udscore))
        print("  Total (params={:,}, flops={:,})".format(total_params, total_flops))
        print("  {}".format("-" * num_udscore))

    return total_params, total_flops
