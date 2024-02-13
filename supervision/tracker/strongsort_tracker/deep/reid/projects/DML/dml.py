from __future__ import absolute_import, division, print_function

import torch
from torch.nn import functional as F

from supervision.tracker.strongsort_tracker.deep.reid.torchreid.engine import Engine
from supervision.tracker.strongsort_tracker.deep.reid.torchreid.losses import (
    CrossEntropyLoss,
    TripletLoss,
)
from supervision.tracker.strongsort_tracker.deep.reid.torchreid.utils import (
    open_all_layers,
    open_specified_layers,
)


class ImageDMLEngine(Engine):
    def __init__(
        self,
        datamanager,
        model1,
        optimizer1,
        scheduler1,
        model2,
        optimizer2,
        scheduler2,
        margin=0.3,
        weight_t=0.5,
        weight_x=1.0,
        weight_ml=1.0,
        use_gpu=True,
        label_smooth=True,
        deploy="model1",
    ):
        super(ImageDMLEngine, self).__init__(datamanager, use_gpu)

        self.model1 = model1
        self.optimizer1 = optimizer1
        self.scheduler1 = scheduler1
        self.register_model("model1", model1, optimizer1, scheduler1)

        self.model2 = model2
        self.optimizer2 = optimizer2
        self.scheduler2 = scheduler2
        self.register_model("model2", model2, optimizer2, scheduler2)

        self.weight_t = weight_t
        self.weight_x = weight_x
        self.weight_ml = weight_ml

        assert deploy in ["model1", "model2", "both"]
        self.deploy = deploy

        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth,
        )

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        outputs1, features1 = self.model1(imgs)
        loss1_x = self.compute_loss(self.criterion_x, outputs1, pids)
        loss1_t = self.compute_loss(self.criterion_t, features1, pids)

        outputs2, features2 = self.model2(imgs)
        loss2_x = self.compute_loss(self.criterion_x, outputs2, pids)
        loss2_t = self.compute_loss(self.criterion_t, features2, pids)

        loss1_ml = self.compute_kl_div(outputs2.detach(), outputs1, is_logit=True)
        loss2_ml = self.compute_kl_div(outputs1.detach(), outputs2, is_logit=True)

        loss1 = 0
        loss1 += loss1_x * self.weight_x
        loss1 += loss1_t * self.weight_t
        loss1 += loss1_ml * self.weight_ml

        loss2 = 0
        loss2 += loss2_x * self.weight_x
        loss2 += loss2_t * self.weight_t
        loss2 += loss2_ml * self.weight_ml

        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step()

        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer2.step()

        loss_dict = {
            "loss1_x": loss1_x.item(),
            "loss1_t": loss1_t.item(),
            "loss1_ml": loss1_ml.item(),
            "loss2_x": loss1_x.item(),
            "loss2_t": loss1_t.item(),
            "loss2_ml": loss1_ml.item(),
        }

        return loss_dict

    @staticmethod
    def compute_kl_div(p, q, is_logit=True):
        if is_logit:
            p = F.softmax(p, dim=1)
            q = F.softmax(q, dim=1)
        return -(p * torch.log(q + 1e-8)).sum(1).mean()

    def two_stepped_transfer_learning(
        self, epoch, fixbase_epoch, open_layers, model=None
    ):
        """Two stepped transfer learning.

        The idea is to freeze base layers for a certain number of epochs
        and then open all layers for training.

        Reference: https://arxiv.org/abs/1611.05244
        """
        model1 = self.model1
        model2 = self.model2

        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print(
                "* Only train {} (epoch: {}/{})".format(
                    open_layers, epoch + 1, fixbase_epoch
                )
            )
            open_specified_layers(model1, open_layers)
            open_specified_layers(model2, open_layers)
        else:
            open_all_layers(model1)
            open_all_layers(model2)

    def extract_features(self, input):
        if self.deploy == "model1":
            return self.model1(input)

        elif self.deploy == "model2":
            return self.model2(input)

        else:
            features = []
            features.append(self.model1(input))
            features.append(self.model2(input))
            return torch.cat(features, 1)
