import numpy as np

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker
from supervision.detection.core import Detections
import os
import gdown

__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path=None, max_dist=0.2, use_cuda=True):
        

        self._default_model_dir = os.path.join(os.getcwd(), "supervision", "tracker", "deep_sort", "weights")
        if model_path is None:
            if not(os.path.isfile(os.path.join(self._default_model_dir, "ckpt.t7"))):
                os.makedirs(self._default_model_dir, exist_ok=True)
                
                gdown.download(
                    "https://drive.google.com/uc?id=1_qwTWdzT9dWNudpusgKavj_4elGgbkUN", 
                    os.path.join(self._default_model_dir, "ckpt.t7"), 
                    quiet=False
                    )
        else:
            self._default_model_dir = model_path

        model_path = os.path.join(self._default_model_dir, "ckpt.t7")


        

        self.min_confidence = 0.3
        self.nms_max_overlap = 1.0

        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

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
            detections.class_id = np.array(
                [int(t[5]) for t in tracks], dtype=int
            )
            detections.tracker_id = np.array(
                [int(t[4]) for t in tracks], dtype=int
            )
            detections.confidence = np.array(
                [t[-1] for t in tracks], dtype=np.float32
            )
        else:
            detections.tracker_id = np.array([], dtype=int)

        return detections



    @staticmethod
    def _convert_xyxy_tlwh(bbox_xyxy):
        bbox_tlwh = []

        for bbox in bbox_xyxy:
            bbox = [int(e) for e in bbox]
            tlx,tly,brx,bry = bbox

            w = brx - tlx
            h = bry - tly
            bbox_tlwh.append([tlx, tly, w, h])

        return bbox_tlwh

    def update(self, bbox_xyxy, confidences, class_ids, ori_img):
        self.height, self.width = ori_img.shape[:2]

        features = self._get_features(bbox_xyxy, ori_img)
        bbox_tlwh = self._convert_xyxy_tlwh(bbox_xyxy)
        
        # bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i,conf in enumerate(confidences)] #if conf>self.min_confidence]
        
        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = [d.confidence for d in detections]
        indices = non_max_suppression(boxes, self.nms_max_overlap, np.array(scores))
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)
        # output bbox identities
        outputs = []

        for index, track in enumerate(self.tracker.tracks):
            if index >= len(scores):
                continue

            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append([x1,y1,x2,y2,track_id, class_ids[index], scores[index]])
        # if len(outputs) > 0:
        #     outputs = np.stack(outputs,axis=0)
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        bbox_xywh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_xywh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_xywh


    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2
    
    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            # x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            box = [int(e) for e in box]
            x1,y1,x2,y2 = box
            im = ori_img[y1:y2,x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features


