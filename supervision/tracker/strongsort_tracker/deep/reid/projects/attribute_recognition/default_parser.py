from __future__ import print_function, absolute_import
import argparse


def init_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ************************************************************
    # Datasets
    # ************************************************************
    parser.add_argument(
        '--root',
        type=str,
        default='',
        required=True,
        help='root path to data directory'
    )
    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        required=True,
        help='which dataset to choose'
    )
    parser.add_argument(
        '-j',
        '--workers',
        type=int,
        default=4,
        help='number of data loading workers (tips: 4 or 8 times number of gpus)'
    )
    parser.add_argument(
        '--height', type=int, default=256, help='height of an image'
    )
    parser.add_argument(
        '--width', type=int, default=128, help='width of an image'
    )

    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument(
        '--optim',
        type=str,
        default='adam',
        help='optimization algorithm (see optimizers.py)'
    )
    parser.add_argument(
        '--lr', type=float, default=0.0003, help='initial learning rate'
    )
    parser.add_argument(
        '--weight-decay', type=float, default=5e-04, help='weight decay'
    )
    # sgd
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='momentum factor for sgd and rmsprop'
    )
    parser.add_argument(
        '--sgd-dampening',
        type=float,
        default=0,
        help='sgd\'s dampening for momentum'
    )
    parser.add_argument(
        '--sgd-nesterov',
        action='store_true',
        help='whether to enable sgd\'s Nesterov momentum'
    )
    # rmsprop
    parser.add_argument(
        '--rmsprop-alpha',
        type=float,
        default=0.99,
        help='rmsprop\'s smoothing constant'
    )
    # adam/amsgrad
    parser.add_argument(
        '--adam-beta1',
        type=float,
        default=0.9,
        help='exponential decay rate for adam\'s first moment'
    )
    parser.add_argument(
        '--adam-beta2',
        type=float,
        default=0.999,
        help='exponential decay rate for adam\'s second moment'
    )

    # ************************************************************
    # Training hyperparameters
    # ************************************************************
    parser.add_argument(
        '--max-epoch', type=int, default=60, help='maximum epochs to run'
    )
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='manual epoch number (useful when restart)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32, help='batch size'
    )

    parser.add_argument(
        '--fixbase-epoch',
        type=int,
        default=0,
        help='number of epochs to fix base layers'
    )
    parser.add_argument(
        '--open-layers',
        type=str,
        nargs='+',
        default=['classifier'],
        help='open specified layers for training while keeping others frozen'
    )

    parser.add_argument(
        '--staged-lr',
        action='store_true',
        help='set different lr to different layers'
    )
    parser.add_argument(
        '--new-layers',
        type=str,
        nargs='+',
        default=['classifier'],
        help='newly added layers with default lr'
    )
    parser.add_argument(
        '--base-lr-mult',
        type=float,
        default=0.1,
        help='learning rate multiplier for base layers'
    )

    # ************************************************************
    # Learning rate scheduler options
    # ************************************************************
    parser.add_argument(
        '--lr-scheduler',
        type=str,
        default='multi_step',
        help='learning rate scheduler (see lr_schedulers.py)'
    )
    parser.add_argument(
        '--stepsize',
        type=int,
        default=[20, 40],
        nargs='+',
        help='stepsize to decay learning rate'
    )
    parser.add_argument(
        '--gamma', type=float, default=0.1, help='learning rate decay'
    )

    # ************************************************************
    # Architecture
    # ************************************************************
    parser.add_argument(
        '-a', '--arch', type=str, default='', help='model architecture'
    )
    parser.add_argument(
        '--no-pretrained',
        action='store_true',
        help='do not load pretrained weights'
    )

    # ************************************************************
    # Loss
    # ************************************************************
    parser.add_argument(
        '--weighted-bce', action='store_true', help='use weighted BCELoss'
    )

    # ************************************************************
    # Test settings
    # ************************************************************
    parser.add_argument(
        '--load-weights', type=str, default='', help='load pretrained weights'
    )
    parser.add_argument(
        '--evaluate', action='store_true', help='evaluate only'
    )
    parser.add_argument(
        '--save-prediction', action='store_true', help='save prediction'
    )

    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument(
        '--print-freq', type=int, default=20, help='print frequency'
    )
    parser.add_argument('--seed', type=int, default=1, help='manual seed')
    parser.add_argument(
        '--resume',
        type=str,
        default='',
        metavar='PATH',
        help='resume from a checkpoint'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='log',
        help='path to save log and model weights'
    )
    parser.add_argument('--use-cpu', action='store_true', help='use cpu')

    return parser


def optimizer_kwargs(parsed_args):
    return {
        'optim': parsed_args.optim,
        'lr': parsed_args.lr,
        'weight_decay': parsed_args.weight_decay,
        'momentum': parsed_args.momentum,
        'sgd_dampening': parsed_args.sgd_dampening,
        'sgd_nesterov': parsed_args.sgd_nesterov,
        'rmsprop_alpha': parsed_args.rmsprop_alpha,
        'adam_beta1': parsed_args.adam_beta1,
        'adam_beta2': parsed_args.adam_beta2,
        'staged_lr': parsed_args.staged_lr,
        'new_layers': parsed_args.new_layers,
        'base_lr_mult': parsed_args.base_lr_mult
    }


def lr_scheduler_kwargs(parsed_args):
    return {
        'lr_scheduler': parsed_args.lr_scheduler,
        'stepsize': parsed_args.stepsize,
        'gamma': parsed_args.gamma
    }
