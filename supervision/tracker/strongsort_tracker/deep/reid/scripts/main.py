import argparse
import os.path as osp
import sys
import time

import torch
import torch.nn as nn
import torchreid
from default_config import (
    engine_run_kwargs,
    get_default_config,
    imagedata_kwargs,
    lr_scheduler_kwargs,
    optimizer_kwargs,
    videodata_kwargs,
)

from supervision.tracker.strongsort_tracker.deep.reid.torchreid.utils import (
    Logger,
    check_isfile,
    collect_env_info,
    compute_model_complexity,
    load_pretrained_weights,
    resume_from_checkpoint,
    set_random_seed,
)


def build_datamanager(cfg):
    if cfg.data.type == "image":
        return torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))


def build_engine(cfg, datamanager, model, optimizer, scheduler):
    if cfg.data.type == "image":
        if cfg.loss.name == "softmax":
            engine = torchreid.engine.ImageSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
            )

        else:
            engine = torchreid.engine.ImageTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
            )

    else:
        if cfg.loss.name == "softmax":
            engine = torchreid.engine.VideoSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                pooling_method=cfg.video.pooling_method,
            )

        else:
            engine = torchreid.engine.VideoTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
            )

    return engine


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms


def check_cfg(cfg):
    if cfg.loss.name == "triplet" and cfg.loss.triplet.weight_x == 0:
        assert (
            cfg.train.fixbase_epoch == 0
        ), "The output of classifier is not included in the computational graph"


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "-s",
        "--sources",
        type=str,
        nargs="+",
        help="source datasets (delimited by space)",
    )
    parser.add_argument(
        "-t",
        "--targets",
        type=str,
        nargs="+",
        help="target datasets (delimited by space)",
    )
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation")
    parser.add_argument("--root", type=str, default="", help="path to data root")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    set_random_seed(cfg.train.seed)
    check_cfg(cfg)

    log_name = "test.log" if cfg.test.evaluate else "train.log"
    log_name += time.strftime("-%Y-%m-%d-%H-%M-%S")
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

    print("Show configuration\n{}\n".format(cfg))
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    datamanager = build_datamanager(cfg)

    print("Building model: {}".format(cfg.model.name))
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu,
    )
    num_params, flops = compute_model_complexity(
        model, (1, 3, cfg.data.height, cfg.data.width)
    )
    print("Model complexity: params={:,} flops={:,}".format(num_params, flops))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()

    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, **lr_scheduler_kwargs(cfg)
    )

    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
        )

    print("Building {}-engine for {}-reid".format(cfg.loss.name, cfg.data.type))
    engine = build_engine(cfg, datamanager, model, optimizer, scheduler)
    engine.run(**engine_run_kwargs(cfg))


if __name__ == "__main__":
    main()
