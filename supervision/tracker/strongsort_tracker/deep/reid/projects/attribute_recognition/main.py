from __future__ import division, print_function
import sys
import copy
import time
import numpy as np
import os.path as osp
import datetime
import warnings
import torch
import torch.nn as nn

import torchreid
from supervision.tracker.strongsort_tracker.deep.reid.torchreid.utils import (
    Logger, AverageMeter, check_isfile, open_all_layers, save_checkpoint,
    set_random_seed, collect_env_info, open_specified_layers,
    load_pretrained_weights, compute_model_complexity
)
from supervision.tracker.strongsort_tracker.deep.reid.torchreid.data.transforms import (
    Resize, Compose, ToTensor, Normalize, Random2DTranslation,
    RandomHorizontalFlip
)

import models
import datasets
from default_parser import init_parser, optimizer_kwargs, lr_scheduler_kwargs

parser = init_parser()
args = parser.parse_args()


def init_dataset(use_gpu):
    normalize = Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform_tr = Compose(
        [
            Random2DTranslation(args.height, args.width, p=0.5),
            RandomHorizontalFlip(),
            ToTensor(), normalize
        ]
    )

    transform_te = Compose(
        [Resize([args.height, args.width]),
         ToTensor(), normalize]
    )

    trainset = datasets.init_dataset(
        args.dataset,
        root=args.root,
        transform=transform_tr,
        mode='train',
        verbose=True
    )

    valset = datasets.init_dataset(
        args.dataset,
        root=args.root,
        transform=transform_te,
        mode='val',
        verbose=False
    )

    testset = datasets.init_dataset(
        args.dataset,
        root=args.root,
        transform=transform_te,
        mode='test',
        verbose=False
    )

    num_attrs = trainset.num_attrs
    attr_dict = trainset.attr_dict

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=use_gpu,
        drop_last=True
    )

    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=use_gpu,
        drop_last=False
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=use_gpu,
        drop_last=False
    )

    return trainloader, valloader, testloader, num_attrs, attr_dict


def main():
    global args

    set_random_seed(args.seed)
    use_gpu = torch.cuda.is_available() and not args.use_cpu
    log_name = 'test.log' if args.evaluate else 'train.log'
    sys.stdout = Logger(osp.join(args.save_dir, log_name))

    print('** Arguments **')
    arg_keys = list(args.__dict__.keys())
    arg_keys.sort()
    for key in arg_keys:
        print('{}: {}'.format(key, args.__dict__[key]))
    print('\n')
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    if use_gpu:
        torch.backends.cudnn.benchmark = True
    else:
        warnings.warn(
            'Currently using CPU, however, GPU is highly recommended'
        )

    dataset_vars = init_dataset(use_gpu)
    trainloader, valloader, testloader, num_attrs, attr_dict = dataset_vars

    if args.weighted_bce:
        print('Use weighted binary cross entropy')
        print('Computing the weights ...')
        bce_weights = torch.zeros(num_attrs, dtype=torch.float)
        for _, attrs, _ in trainloader:
            bce_weights += attrs.sum(0) # sum along the batch dim
        bce_weights /= len(trainloader) * args.batch_size
        print('Sample ratio for each attribute: {}'.format(bce_weights))
        bce_weights = torch.exp(-1 * bce_weights)
        print('BCE weights: {}'.format(bce_weights))
        bce_weights = bce_weights.expand(args.batch_size, num_attrs)
        criterion = nn.BCEWithLogitsLoss(weight=bce_weights)

    else:
        print('Use plain binary cross entropy')
        criterion = nn.BCEWithLogitsLoss()

    print('Building model: {}'.format(args.arch))
    model = models.build_model(
        args.arch,
        num_attrs,
        pretrained=not args.no_pretrained,
        use_gpu=use_gpu
    )
    num_params, flops = compute_model_complexity(
        model, (1, 3, args.height, args.width)
    )
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)

    if use_gpu:
        model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()

    if args.evaluate:
        test(model, testloader, attr_dict, use_gpu)
        return

    optimizer = torchreid.optim.build_optimizer(
        model, **optimizer_kwargs(args)
    )
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, **lr_scheduler_kwargs(args)
    )

    start_epoch = args.start_epoch
    best_result = -np.inf
    if args.resume and check_isfile(args.resume):
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_result = checkpoint['label_mA']
        print('Loaded checkpoint from "{}"'.format(args.resume))
        print('- start epoch: {}'.format(start_epoch))
        print('- label_mA: {}'.format(best_result))

    time_start = time.time()

    for epoch in range(start_epoch, args.max_epoch):
        train(
            epoch, model, criterion, optimizer, scheduler, trainloader, use_gpu
        )
        test_outputs = test(model, testloader, attr_dict, use_gpu)
        label_mA = test_outputs[0]
        is_best = label_mA > best_result
        if is_best:
            best_result = label_mA

        save_checkpoint(
            {
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'label_mA': label_mA,
                'optimizer': optimizer.state_dict(),
            },
            args.save_dir,
            is_best=is_best
        )

    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))


def train(epoch, model, criterion, optimizer, scheduler, trainloader, use_gpu):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.train()

    if (epoch + 1) <= args.fixbase_epoch and args.open_layers is not None:
        print(
            '* Only train {} (epoch: {}/{})'.format(
                args.open_layers, epoch + 1, args.fixbase_epoch
            )
        )
        open_specified_layers(model, args.open_layers)
    else:
        open_all_layers(model)

    end = time.time()
    for batch_idx, data in enumerate(trainloader):
        data_time.update(time.time() - end)

        imgs, attrs = data[0], data[1]
        if use_gpu:
            imgs = imgs.cuda()
            attrs = attrs.cuda()

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, attrs)
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        losses.update(loss.item(), imgs.size(0))

        if (batch_idx+1) % args.print_freq == 0:
            # estimate remaining time
            num_batches = len(trainloader)
            eta_seconds = batch_time.avg * (
                num_batches - (batch_idx+1) + (args.max_epoch -
                                               (epoch+1)) * num_batches
            )
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            print(
                'Epoch: [{0}/{1}][{2}/{3}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Lr {lr:.6f}\t'
                'Eta {eta}'.format(
                    epoch + 1,
                    args.max_epoch,
                    batch_idx + 1,
                    len(trainloader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.param_groups[0]['lr'],
                    eta=eta_str
                )
            )

        end = time.time()

    scheduler.step()


@torch.no_grad()
def test(model, testloader, attr_dict, use_gpu):
    batch_time = AverageMeter()
    model.eval()

    num_persons = 0
    prob_thre = 0.5
    ins_acc = 0
    ins_prec = 0
    ins_rec = 0
    mA_history = {
        'correct_pos': 0,
        'real_pos': 0,
        'correct_neg': 0,
        'real_neg': 0
    }

    print('Testing ...')

    for batch_idx, data in enumerate(testloader):
        imgs, attrs, img_paths = data
        if use_gpu:
            imgs = imgs.cuda()

        end = time.time()
        orig_outputs = model(imgs)
        batch_time.update(time.time() - end)

        orig_outputs = orig_outputs.data.cpu().numpy()
        attrs = attrs.data.numpy()

        # transform raw outputs to attributes (binary codes)
        outputs = copy.deepcopy(orig_outputs)
        outputs[outputs < prob_thre] = 0
        outputs[outputs >= prob_thre] = 1

        # compute label-based metric
        overlaps = outputs * attrs
        mA_history['correct_pos'] += overlaps.sum(0)
        mA_history['real_pos'] += attrs.sum(0)
        inv_overlaps = (1-outputs) * (1-attrs)
        mA_history['correct_neg'] += inv_overlaps.sum(0)
        mA_history['real_neg'] += (1 - attrs).sum(0)

        outputs = outputs.astype(bool)
        attrs = attrs.astype(bool)

        # compute instabce-based accuracy
        intersect = (outputs & attrs).astype(float)
        union = (outputs | attrs).astype(float)
        ins_acc += (intersect.sum(1) / union.sum(1)).sum()
        ins_prec += (intersect.sum(1) / outputs.astype(float).sum(1)).sum()
        ins_rec += (intersect.sum(1) / attrs.astype(float).sum(1)).sum()

        num_persons += imgs.size(0)

        if (batch_idx+1) % args.print_freq == 0:
            print(
                'Processed batch {}/{}'.format(batch_idx + 1, len(testloader))
            )

        if args.save_prediction:
            txtfile = open(osp.join(args.save_dir, 'prediction.txt'), 'a')
            for idx in range(imgs.size(0)):
                img_path = img_paths[idx]
                probs = orig_outputs[idx, :]
                labels = attrs[idx, :]
                txtfile.write('{}\n'.format(img_path))
                txtfile.write('*** Correct prediction ***\n')
                for attr_idx, (label, prob) in enumerate(zip(labels, probs)):
                    if label:
                        attr_name = attr_dict[attr_idx]
                        info = '{}: {:.1%}  '.format(attr_name, prob)
                        txtfile.write(info)
                txtfile.write('\n*** Incorrect prediction ***\n')
                for attr_idx, (label, prob) in enumerate(zip(labels, probs)):
                    if not label and prob > 0.5:
                        attr_name = attr_dict[attr_idx]
                        info = '{}: {:.1%}  '.format(attr_name, prob)
                        txtfile.write(info)
                txtfile.write('\n\n')
            txtfile.close()

    print(
        '=> BatchTime(s)/BatchSize(img): {:.4f}/{}'.format(
            batch_time.avg, args.batch_size
        )
    )

    ins_acc /= num_persons
    ins_prec /= num_persons
    ins_rec /= num_persons
    ins_f1 = (2*ins_prec*ins_rec) / (ins_prec+ins_rec)

    term1 = mA_history['correct_pos'] / mA_history['real_pos']
    term2 = mA_history['correct_neg'] / mA_history['real_neg']
    label_mA_verbose = (term1+term2) * 0.5
    label_mA = label_mA_verbose.mean()

    print('* Results *')
    print('  # test persons: {}'.format(num_persons))
    print('  (instance-based)  accuracy:      {:.1%}'.format(ins_acc))
    print('  (instance-based)  precition:     {:.1%}'.format(ins_prec))
    print('  (instance-based)  recall:        {:.1%}'.format(ins_rec))
    print('  (instance-based)  f1-score:      {:.1%}'.format(ins_f1))
    print('  (label-based)     mean accuracy: {:.1%}'.format(label_mA))
    print('  mA for each attribute: {}'.format(label_mA_verbose))

    return label_mA, ins_acc, ins_prec, ins_rec, ins_f1


if __name__ == '__main__':
    main()
