import gdown
from pathlib import Path
import torch 
import yaml 
from types import SimpleNamespace
import os 
from supervision.tracker.strongsort_tracker.strong_sort import StrongSORT


def load_strong_sort():
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

    return strongsort