import numpy as np
from tqdm.notebook import tqdm
from ultralytics import YOLO

import supervision as sv
from supervision import ByteTrack, detections2boxes, match_detections_with_tracks
from supervision.detection.annotate import BoxAnnotator
from supervision.utils.video import VideoInfo, VideoSink, get_video_frames_generator

model = YOLO("yolov5s.pt")
CLASS_NAMES_DICT = model.model.names
SOURCE_VIDEO_PATH = "walking.mp4"
TARGET_VIDEO_PATH = "output.mp4"

byte_tracker = ByteTrack(
    track_thresh=0.25,
    track_buffer=30,
    match_thresh=0.8,
    aspect_ratio_thresh=3.0,
    min_box_area=1.0,
    frame_rate=30,
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
        detections = sv.Detections.from_yolov8(results[0])

        # update tracker
        tracks = byte_tracker.update_from_numpy(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape,
        )

        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        if tracker_id != None:
            detections.tracker_id = np.array(tracker_id)
        frame_text_list = []
        if len(detections) > 0:
            for res in detections:
                confidence = res[2]
                class_id = res[3]
                tracker_id = res[4]
                frame_text = (
                    f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                )
                frame_text_list.append(frame_text)
        # draw bbox and add class name with track id
        frame = annotator.annotate(
            scene=frame, detections=detections, labels=frame_text_list
        )
        sink.write_frame(frame)
