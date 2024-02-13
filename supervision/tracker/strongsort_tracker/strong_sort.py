import os
from os.path import exists as file_exists
from pathlib import Path
from types import SimpleNamespace

import gdown
import numpy as np
import torch
import yaml

from supervision.detection.core import Detections
from supervision.tracker.strongsort_tracker.deep.reid.torchreid.utils import (
    FeatureExtractor,
)

from .deep.reid_model_factory import (
    get_model_name,
    get_model_url,
    show_downloadeable_models,
)
from .sort.detection import Detection
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.tracker import Tracker

__all__ = ["StrongSort"]


class StrongSort(object):
    def __init__(
        self,
        device=None,
        max_dist=0.2,
        max_iou_distance=0.7,
        max_age=70,
        n_init=3,
        nn_budget=100,
        mc_lambda=0.995,
        ema_alpha=0.9,
    ):
        (
            model_weights,
            dev,
            max_dist,
            max_iou_distance,
            max_age,
            n_init,
            nn_budget,
            mc_lambda,
            ema_alpha,
        ) = self.load_config()
        if device == None:
            device = dev
        model_name = get_model_name(model_weights)
        model_url = get_model_url(model_weights)

        if not file_exists(model_weights) and model_url is not None:
            gdown.download(model_url, str(model_weights), quiet=False)
        elif file_exists(model_weights):
            pass
        elif model_url is None:
            print("No URL associated to the chosen StrongSort weights. Choose between:")
            show_downloadeable_models()
            exit()

        self.extractor = FeatureExtractor(
            model_name=model_name, model_path=model_weights, device=str(device)
        )

        self.max_dist = max_dist
        metric = NearestNeighborDistanceMetric("cosine", self.max_dist, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init
        )

    def load_config(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        half = False
        tracker_config = (
            "./supervision/tracker/strongsort_tracker/configs/strong_sort.yaml"
        )
        with open(tracker_config, "r") as f:
            cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        cfg = SimpleNamespace(**cfg)
        model_url = "https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF"

        os.makedirs("./supervision/tracker/strongsort_tracker/weights", exist_ok=True)
        reid_weights = "./supervision/tracker/strongsort_tracker/weights/osnet_x0_25_msmt17.pt"  ##The suffix of the file name is pt
        if not os.path.exists(reid_weights):
            gdown.download(model_url, str(reid_weights), quiet=False)
        reid_weights = Path(reid_weights)

        return (
            reid_weights,
            device,
            cfg.max_dist,
            cfg.max_iou_dist,
            cfg.max_age,
            cfg.n_init,
            cfg.nn_budget,
            cfg.mc_lambda,
            cfg.ema_alpha,
        )

    def update_with_detections(self, detections, ori_img):
        bbox_xywh = detections.xyxy
        confidences = detections.confidence
        class_ids = detections.class_id

        tracks = self.update(bbox_xywh, confidences, class_ids, ori_img)

        detections = Detections.empty()
        if len(tracks) > 0:
            detections.xyxy = np.array(
                [track[0:4] for track in tracks], dtype=np.float32
            )
            detections.class_id = np.array([int(t[5]) for t in tracks], dtype=int)
            detections.tracker_id = np.array([int(t[4]) for t in tracks], dtype=int)
            detections.confidence = np.array([t[-1] for t in tracks], dtype=np.float32)
        else:
            detections.tracker_id = np.array([], dtype=int)

        return detections

    @staticmethod
    def _convert_xyxy_tlwh(bbox_xyxy):
        bbox_tlwh = []

        for bbox in bbox_xyxy:
            bbox = [int(e) for e in bbox]
            tlx, tly, brx, bry = bbox

            w = brx - tlx
            h = bry - tly
            bbox_tlwh.append([tlx, tly, w, h])

        return bbox_tlwh

    def update(self, bbox_xywh, confidences, classes, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._convert_xyxy_tlwh(bbox_xywh)
        detections = [
            Detection(bbox_tlwh[i], conf, features[i])
            for i, conf in enumerate(confidences)
        ]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes, confidences)

        # output bbox identities
        outputs = []
        for index, track in enumerate(self.tracker.tracks):
            if index >= len(scores):
                continue
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)

            track_id = track.track_id
            class_id = track.class_id
            conf = track.conf
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf]))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.0
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.0
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features
