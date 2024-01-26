import os 
import yaml
import torch 
import numpy as np
from types import SimpleNamespace
import numpy as np
from boxmot.appearance.reid_multibackend import ReIDDetectMultiBackend
from boxmot.motion.cmc import get_cmc_method
from boxmot.trackers.strongsort.sort.detection import Detection
from boxmot.trackers.strongsort.sort.tracker import Tracker
from boxmot.utils.matching import NearestNeighborDistanceMetric
from boxmot.utils.ops import xyxy2tlwh


class StrongSORT(object):
    def __init__(
        self,
        model_weights,
        device,
        fp16,
        max_dist=0.2,
        max_iou_dist=0.7,
        max_age=30,
        n_init=1,
        nn_budget=100,
        mc_lambda=0.995,
        ema_alpha=0.9,
    ):
        self.model = ReIDDetectMultiBackend(
            weights=model_weights, device=device, fp16=fp16
        )
        self.tracker = Tracker(
            metric=NearestNeighborDistanceMetric("cosine", max_dist, nn_budget),
            max_iou_dist=max_iou_dist,
            max_age=max_age,
            n_init=n_init,
            mc_lambda=mc_lambda,
            ema_alpha=ema_alpha,
        )
        self.cmc = get_cmc_method("ecc")()

    def update(self, dets, img):
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img' input format '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        xyxy = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]
        det_ind = dets[:, 6]

        if len(self.tracker.tracks) >= 1:
            warp_matrix = self.cmc.apply(img, xyxy)
            for track in self.tracker.tracks:
                track.camera_update(warp_matrix)

        # extract appearance information for each detection
        features = self.model.get_features(xyxy, img)

        tlwh = xyxy2tlwh(xyxy)
        detections = [
            Detection(box, conf, cls, det_ind, feat)
            for box, conf, cls, det_ind, feat in zip(
                tlwh, confs, clss, det_ind, features
            )
        ]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update >= 1:
                continue

            x1, y1, x2, y2 = track.to_tlbr()

            id = track.id
            conf = track.conf
            cls = track.cls
            det_ind = track.det_ind

            outputs.append(
                np.concatenate(
                    ([x1, y1, x2, y2], [id], [conf], [cls], [det_ind])
                ).reshape(1, -1)
            )
        if len(outputs) > 0:
            return np.concatenate(outputs)
        return np.array([])
    


if __name__ == "__main__":
    import gdown
    from pathlib import Path

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half         = False 
    tracker_config = "./supervision/tracker/strongsort_tracker/strongsort.yaml"
    with open(tracker_config, "r") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg = SimpleNamespace(**cfg)  # easier dict acces by dot, instead of ['']

    model_url = "https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF"  
    reid_weights = "./supervision/tracker/strongsort_tracker/weights/osnet_x0_25_msmt17.pt"  ##The suffix of the file name is pt
    if not os.path.exists(reid_weights):
        gdown.download(model_url, str(reid_weights), quiet=False)

    reid_weights = Path(reid_weights)

    strongsort = StrongSORT(
        reid_weights,
        device,
        half,
        max_dist=cfg.max_dist,
        max_iou_dist=cfg.max_iou_dist,
        max_age=cfg.max_age,
        n_init=cfg.n_init,
        nn_budget=cfg.nn_budget,
        mc_lambda=cfg.mc_lambda,
        ema_alpha=cfg.ema_alpha,

    )

    
