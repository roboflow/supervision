from collections import defaultdict, deque

import numpy as np

import supervision as sv
from supervision import _cv2 as cv2

SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

VEHICLE_CLASS_IDS = [3, 4, 6, 8]


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        """Build a perspective transform from image to ground-plane points."""
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Project image points onto the configured ground plane."""
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


def calculate_speed(distance: float, elapsed_frames: int, fps: float) -> float:
    """Convert displacement over source-frame intervals to kilometres per hour."""
    if elapsed_frames < 1:
        raise ValueError("At least one elapsed frame is required to calculate speed.")
    elapsed_time = elapsed_frames / fps
    return distance / elapsed_time * 3.6


def main(
    source_video_path: str,
    target_video_path: str | None = None,
    device: str = "cpu",
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
) -> None:
    """
    Vehicle Speed Estimation using RF-DETR and Supervision.

    Args:
        source_video_path: Path to the source video file
        target_video_path: Path to the target video file (output)
        device: Computation device ('cpu', 'mps' or 'cuda')
        confidence_threshold: Confidence threshold for the model
        iou_threshold: IOU threshold for the model
    """
    from rfdetr import RFDETRMedium

    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    model = RFDETRMedium(device=device)

    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_activation_threshold=confidence_threshold
    )

    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=int(video_info.fps * 2),
        position=sv.Position.BOTTOM_CENTER,
    )

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=int(video_info.fps)))

    def process_frame(frame: np.ndarray, frame_index: int) -> np.ndarray:
        """Detect vehicles, estimate their speeds, and annotate one BGR frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = model.predict(frame_rgb, threshold=confidence_threshold)
        detections = detections[np.isin(detections.class_id, VEHICLE_CLASS_IDS)]
        detections = detections.with_nms(threshold=iou_threshold)
        detections = detections[polygon_zone.trigger(detections)]
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)
        for tracker_id, [_, y] in zip(detections.tracker_id, points, strict=True):
            coordinates[tracker_id].append((frame_index, y))

        labels = []
        for tracker_id in detections.tracker_id:
            history = coordinates[tracker_id]
            elapsed_frames = history[-1][0] - history[0][0]
            if len(history) < 2 or elapsed_frames < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
                continue
            distance = abs(history[-1][1] - history[0][1])
            speed = calculate_speed(distance, elapsed_frames, video_info.fps)
            labels.append(f"#{tracker_id} {int(speed)} km/h")

        annotated_frame = trace_annotator.annotate(
            scene=frame.copy(), detections=detections
        )
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        return label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

    if target_video_path is not None:
        with sv.VideoSink(target_video_path, video_info) as sink:
            for frame_index, frame in enumerate(frame_generator):
                sink.write_frame(process_frame(frame, frame_index))
        return

    window = sv.ImageWindow("frame")
    for frame_index, frame in enumerate(frame_generator):
        annotated_frame = process_frame(frame, frame_index)
        window.show(annotated_frame)
        key = window.wait_key(1)
        if not window.is_open or key == "q":
            break
    window.close()


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
