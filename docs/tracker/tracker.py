
from typing import List
import numpy as np
from onemetric.cv.utils.iou import box_iou_batch
from tqdm.notebook import tqdm

from ultralytics import YOLO

from supervision.tracker.byte_tracker.byte_tracker import byte_tracker, strack
from supervision.detection.annotate import BoxAnnotator
from supervision.utils.video import VideoInfo
from supervision.utils.video import get_video_frames_generator
from supervision.utils.video import VideoSink
from supervision import Detections


# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))

# converts List[strack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[strack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)

# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections,
    tracks: List[strack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

model = YOLO('yolov5s.pt')
CLASS_NAMES_DICT  = model.model.names
SOURCE_VIDEO_PATH = "walking.mp4"
TARGET_VIDEO_PATH = "walking_out.mp4"

track_thresh=0.25
track_buffer=30
match_thresh=0.8
aspect_ratio_thresh=3.0
min_box_area=1.0
mot20=False
frame_rate=30

byte_tracker = byte_tracker(track_thresh=0.25, 
                            track_buffer=30, 
                            match_thresh=0.8, 
                            aspect_ratio_thresh=3.0, 
                            min_box_area=1.0, 
                            mot20=False, 
                            frame_rate=30
                            )
annotator = BoxAnnotator()

# create VideoInfo instance
video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
# create frame generator
generator = get_video_frames_generator(SOURCE_VIDEO_PATH)

# open target video file
with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    # loop over video frames
    for frame in tqdm(generator, total=video_info.total_frames):
        # inference
        results = model(frame)
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )

        # update tracker
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )

        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        if tracker_id != None:
            detections.tracker_id = np.array(tracker_id)
        frame_text_list = [] 
        if len(detections) > 0:
            for res in detections:
                confidence = res[2] 
                class_id   = res[3]
                tracker_id = res[4]
                frame_text = f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                frame_text_list.append(frame_text)
        # draw bbox and add class name with track id
        frame = annotator.annotate(scene=frame, detections=detections, labels=frame_text_list)
        sink.write_frame(frame)