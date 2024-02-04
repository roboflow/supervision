from .strong_sort import StrongSort


__all__ = ['StrongSort', 'build_tracker']


def build_tracker(cfg, use_cuda):
    return StrongSort(cfg.StrongSort.REID_CKPT, 
                max_dist=cfg.StrongSort.MAX_DIST, min_confidence=cfg.StrongSort.MIN_CONFIDENCE, 
                nms_max_overlap=cfg.StrongSort.NMS_MAX_OVERLAP, max_iou_distance=cfg.StrongSort.MAX_IOU_DISTANCE, 
                max_age=cfg.StrongSort.MAX_AGE, n_init=cfg.StrongSort.N_INIT, nn_budget=cfg.StrongSort.NN_BUDGET, use_cuda=use_cuda)
