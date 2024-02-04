from __future__ import division, print_function, absolute_import
import torch

from supervision.tracker.strongsort_tracker.deep.reid.torchreid.engine.image import ImageSoftmaxEngine


class VideoSoftmaxEngine(ImageSoftmaxEngine):
    """Softmax-loss engine for video-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.
        pooling_method (str, optional): how to pool features for a tracklet.
            Default is "avg" (average). Choices are ["avg", "max"].

    Examples::
        
        import torch
        import torchreid
        # Each batch contains batch_size*seq_len images
        datamanager = torchreid.data.VideoDataManager(
            root='path/to/reid-data',
            sources='mars',
            height=256,
            width=128,
            combineall=False,
            batch_size=8, # number of tracklets
            seq_len=15 # number of images in each tracklet
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='softmax'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.VideoSoftmaxEngine(
            datamanager, model, optimizer, scheduler=scheduler,
            pooling_method='avg'
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-softmax-mars',
            print_freq=10
        )
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        scheduler=None,
        use_gpu=True,
        label_smooth=True,
        pooling_method='avg'
    ):
        super(VideoSoftmaxEngine, self).__init__(
            datamanager,
            model,
            optimizer,
            scheduler=scheduler,
            use_gpu=use_gpu,
            label_smooth=label_smooth
        )
        self.pooling_method = pooling_method

    def parse_data_for_train(self, data):
        imgs = data['img']
        pids = data['pid']
        if imgs.dim() == 5:
            # b: batch size
            # s: sqeuence length
            # c: channel depth
            # h: height
            # w: width
            b, s, c, h, w = imgs.size()
            imgs = imgs.view(b * s, c, h, w)
            pids = pids.view(b, 1).expand(b, s)
            pids = pids.contiguous().view(b * s)
        return imgs, pids

    def extract_features(self, input):
        # b: batch size
        # s: sqeuence length
        # c: channel depth
        # h: height
        # w: width
        b, s, c, h, w = input.size()
        input = input.view(b * s, c, h, w)
        features = self.model(input)
        features = features.view(b, s, -1)
        if self.pooling_method == 'avg':
            features = torch.mean(features, 1)
        else:
            features = torch.max(features, 1)[0]
        return features
