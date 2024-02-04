import sys
import copy
import time
import os.path as osp
import argparse
import torch
import torch.nn as nn

import torchreid
from supervision.tracker.strongsort_tracker.deep.reid.torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

from dml import ImageDMLEngine
from default_config import (
    imagedata_kwargs, optimizer_kwargs, engine_run_kwargs, get_default_config,
    lr_scheduler_kwargs
)


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '-s',
        '--sources',
        type=str,
        nargs='+',
        help='source datasets (delimited by space)'
    )
    parser.add_argument(
        '-t',
        '--targets',
        type=str,
        nargs='+',
        help='target datasets (delimited by space)'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', help='data augmentation'
    )
    parser.add_argument(
        '--root', type=str, default='', help='path to data root'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='Modify config options using the command-line'
    )
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    set_random_seed(cfg.train.seed)

    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    datamanager = torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))

    print('Building model-1: {}'.format(cfg.model.name))
    model1 = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu
    )
    num_params, flops = compute_model_complexity(
        model1, (1, 3, cfg.data.height, cfg.data.width)
    )
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

    print('Copying model-1 to model-2')
    model2 = copy.deepcopy(model1)

    if cfg.model.load_weights1 and check_isfile(cfg.model.load_weights1):
        load_pretrained_weights(model1, cfg.model.load_weights1)

    if cfg.model.load_weights2 and check_isfile(cfg.model.load_weights2):
        load_pretrained_weights(model2, cfg.model.load_weights2)

    if cfg.use_gpu:
        model1 = nn.DataParallel(model1).cuda()
        model2 = nn.DataParallel(model2).cuda()

    optimizer1 = torchreid.optim.build_optimizer(
        model1, **optimizer_kwargs(cfg)
    )
    scheduler1 = torchreid.optim.build_lr_scheduler(
        optimizer1, **lr_scheduler_kwargs(cfg)
    )

    optimizer2 = torchreid.optim.build_optimizer(
        model2, **optimizer_kwargs(cfg)
    )
    scheduler2 = torchreid.optim.build_lr_scheduler(
        optimizer2, **lr_scheduler_kwargs(cfg)
    )

    if cfg.model.resume1 and check_isfile(cfg.model.resume1):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume1,
            model1,
            optimizer=optimizer1,
            scheduler=scheduler1
        )

    if cfg.model.resume2 and check_isfile(cfg.model.resume2):
        resume_from_checkpoint(
            cfg.model.resume2,
            model2,
            optimizer=optimizer2,
            scheduler=scheduler2
        )

    print('Building DML-engine for image-reid')
    engine = ImageDMLEngine(
        datamanager,
        model1,
        optimizer1,
        scheduler1,
        model2,
        optimizer2,
        scheduler2,
        margin=cfg.loss.triplet.margin,
        weight_t=cfg.loss.triplet.weight_t,
        weight_x=cfg.loss.triplet.weight_x,
        weight_ml=cfg.loss.dml.weight_ml,
        use_gpu=cfg.use_gpu,
        label_smooth=cfg.loss.softmax.label_smooth,
        deploy=cfg.model.deploy
    )
    engine.run(**engine_run_kwargs(cfg))


if __name__ == '__main__':
    main()
