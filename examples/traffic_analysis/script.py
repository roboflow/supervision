import argparse
from typing import Optional, Union

import cv2
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv
import numpy as np

from supervision import Color, ColorPalette, Position, draw_polygon

COLOR = Color.white()

ZONE_POLYGONS = [
    np.array([[592, 282], [900, 282], [900, 82], [592, 82]]),
    np.array([[950, 282], [1250, 282], [1250, 82], [950, 82]]),
    np.array([[950, 860], [1250, 860], [1250, 1060], [950, 1060]]),
    np.array([[592, 860], [900, 860], [900, 1060], [592, 1060]]),
    np.array([[592, 282], [592, 550], [392, 550], [392, 282]]),
    np.array([[592, 582], [592, 860], [392, 860], [392, 582]]),
    np.array([[1250, 282], [1250, 530], [1450, 530], [1450, 282]]),
    np.array([[1250, 860], [1250, 560], [1450, 560], [1450, 860]])
]


class Trace:

    def __init__(
        self,
        max_size: Optional[int] = None,
        start_frame_id: int = 0,
        anchor: sv.Position = sv.Position.CENTER
    ) -> None:
        self.current_frame_id = start_frame_id
        self.max_size = max_size
        self.anchor = anchor

        self.frame_id = np.array([], dtype=int)
        self.xy = np.empty((0, 2), dtype=np.float32)
        self.tracker_id = np.array([], dtype=int)

    def put(self, detections: sv.Detections) -> None:
        frame_id = np.full(len(detections), self.current_frame_id, dtype=int)
        self.frame_id = np.concatenate([self.frame_id, frame_id])
        self.xy = np.concatenate([
            self.xy, detections.get_anchor_coordinates(self.anchor)])
        self.tracker_id = np.concatenate([self.tracker_id, detections.tracker_id])

        unique_frame_id = np.unique(self.frame_id)

        if 0 < self.max_size < len(unique_frame_id):
            max_allowed_frame_id = self.current_frame_id - self.max_size + 1
            filtering_mask = self.frame_id >= max_allowed_frame_id
            self.frame_id = self.frame_id[filtering_mask]
            self.xy = self.xy[filtering_mask]
            self.tracker_id = self.tracker_id[filtering_mask]

        self.current_frame_id += 1

    def get(self, tracker_id: int) -> np.ndarray:
        return self.xy[self.tracker_id == tracker_id]


class TraceAnnotator:

    def __init__(
            self,
            color: Union[Color, ColorPalette] = ColorPalette.default(),
            color_by_track: bool = False,
            position: Optional[Position] = Position.CENTER,
            trace_length: int = 30,
            thickness: int = 2,

    ):
        self.color: Union[Color, ColorPalette] = color
        self.color_by_track = color_by_track
        self.position = position
        self.trace = Trace(max_size=trace_length)
        self.thickness = thickness

    def annotate(self, scene: np.ndarray, detections: sv.Detections) -> np.ndarray:
        self.trace.put(detections)

        for i, (xyxy, mask, confidence, class_id, tracker_id) in enumerate(detections):
            class_id = (
                detections.class_id[i] if class_id is not None else None
            )
            idx = class_id if class_id is not None else i
            color = (
                self.color.by_idx(idx)
                if isinstance(self.color, ColorPalette)
                else self.color
            )

            xy = self.trace.get(tracker_id=tracker_id)
            if len(xy) > 1:
                scene = cv2.polylines(
                    scene,
                    [xy.astype(np.int32)],
                    False,
                    color=color.as_bgr(),
                    thickness=self.thickness,
                )
        return scene


def process_video(
    source_weights_path: str,
    source_video_path: str,
    target_video_path: str = None,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
) -> None:
    model = YOLO(source_weights_path)
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    trace_annotator = TraceAnnotator(trace_length=100)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)

    if target_video_path:
        with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
            for frame in tqdm(frame_generator, total=video_info.total_frames):
                annotated_frame = process_frame(
                    frame, model, tracker, box_annotator,
                    confidence_threshold, iou_threshold, trace_annotator
                )
                sink.write_frame(frame=annotated_frame)
    else:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            annotated_frame = process_frame(
                frame, model, tracker, box_annotator,
                confidence_threshold, iou_threshold, trace_annotator
            )
            cv2.imshow('Processed Video', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


def process_frame(
    frame: np.ndarray,
    model: YOLO,
    tracker: sv.ByteTrack,
    box_annotator: sv.BoxAnnotator,
    confidence_threshold: float,
    iou_threshold: float,
    trace_annotator: TraceAnnotator
) -> np.ndarray:
    results = model(
        frame, verbose=False, conf=confidence_threshold, iou=iou_threshold
    )[0]
    detections = sv.Detections.from_ultralytics(results)
    detections.class_id = np.zeros(len(detections))
    detections = tracker.update_with_detections(detections)
    labels = [
        f"#{tracker_id}"
        for _, _, confidence, class_id, tracker_id in detections
    ]
    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections,
        labels=labels)
    annotated_frame = trace_annotator.annotate(
        scene=annotated_frame,
        detections=detections)

    for zone in ZONE_POLYGONS:
        annotated_frame = draw_polygon(scene=annotated_frame, polygon=zone, color=COLOR)
    return annotated_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video Processing with YOLO and ByteTrack")

    parser.add_argument(
        "--source_weights_path", required=True,
        help="Path to the source weights file", type=str
    )
    parser.add_argument(
        "--source_video_path", required=True,
        help="Path to the source video file", type=str
    )
    parser.add_argument(
        "--target_video_path", default=None,
        help="Path to the target video file (output)", type=str
    )
    parser.add_argument(
        "--confidence_threshold", default=0.3,
        help="Confidence threshold for the model", type=float
    )
    parser.add_argument(
        "--iou_threshold", default=0.7,
        help="IOU threshold for the model", type=float
    )

    args = parser.parse_args()

    process_video(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
