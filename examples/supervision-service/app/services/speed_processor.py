from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from ultralytics import YOLO

import supervision as sv

from app.config import DEFAULT_CONFIDENCE, DEFAULT_IOU, DEFAULT_SPEED_WEIGHTS, OUTPUT_DIR
from app.services.video_encoding import ensure_browser_playable

ProgressCallback = Callable[[int, int], None]


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


def build_target_array(target_width: float, target_height: float) -> np.ndarray:
    """Build the bird's-eye target quadrilateral for perspective transform.

    Args:
        target_width: Real-world width of the calibrated road section in meters.
        target_height: Real-world length of the calibrated road section in meters.

    Returns:
        Target corner coordinates for OpenCV perspective transform.

    Examples:
        >>> arr = build_target_array(25, 250)
        >>> arr.shape
        (4, 2)
    """
    width = max(target_width, 1.0)
    height = max(target_height, 1.0)
    return np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )


def estimate_speed_video(
    source_video_path: Path,
    target_video_path: Path,
    source_points: list[list[int]],
    target_width: float,
    target_height: float,
    weights_path: Path = DEFAULT_SPEED_WEIGHTS,
    confidence_threshold: float = DEFAULT_CONFIDENCE,
    iou_threshold: float = DEFAULT_IOU,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Estimate vehicle speeds and write an annotated output video.

    Args:
        source_video_path: Input video path.
        target_video_path: Output annotated video path.
        source_points: Four [x, y] points defining the road surface quadrilateral.
        target_width: Real-world width of the calibrated section in meters.
        target_height: Real-world length of the calibrated section in meters.
        weights_path: YOLO weights path.
        confidence_threshold: Detection confidence threshold.
        iou_threshold: Detection IOU threshold.

    Returns:
        Path to the annotated output video.

    Examples:
        >>> points = [[0, 0], [100, 0], [100, 100], [0, 100]]
        >>> estimate_speed_video(Path("in.mp4"), Path("out.mp4"), points, 25, 250)  # doctest: +SKIP
        PosixPath('out.mp4')
    """
    if len(source_points) != 4:
        raise ValueError("Exactly four source points are required.")

    source = np.array(source_points, dtype=np.float32)
    target = build_target_array(target_width, target_height)

    video_info = sv.VideoInfo.from_video_path(str(source_video_path))
    model = YOLO(str(weights_path))
    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps,
        track_activation_threshold=confidence_threshold,
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

    frame_generator = sv.get_video_frames_generator(str(source_video_path))
    polygon_zone = sv.PolygonZone(polygon=source)
    view_transformer = ViewTransformer(source=source, target=target)
    total_frames = video_info.total_frames

    coordinates: defaultdict[int, deque[int]] = defaultdict(
        lambda: deque(maxlen=int(video_info.fps))
    )

    target_video_path.parent.mkdir(parents=True, exist_ok=True)

    if on_progress and total_frames:
        on_progress(0, total_frames)

    with sv.VideoSink(str(target_video_path), video_info) as sink:
        for frame_index, frame in enumerate(frame_generator, start=1):
            result = model(
                frame,
                verbose=False,
                conf=confidence_threshold,
                iou=iou_threshold,
            )[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[polygon_zone.trigger(detections)]
            detections = byte_track.update_with_detections(detections=detections)

            points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
            points = view_transformer.transform_points(points=points).astype(int)

            for tracker_id, (_, y) in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)

            labels = []
            for tracker_id in detections.tracker_id:
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id}")
                else:
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time * 3.6
                    labels.append(f"#{tracker_id} {int(speed)} km/h")

            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
            )
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels,
            )
            sink.write_frame(annotated_frame)

            if on_progress and total_frames:
                on_progress(frame_index, total_frames)

    if on_progress and total_frames:
        on_progress(total_frames, total_frames)

    ensure_browser_playable(target_video_path)
    return target_video_path


def build_speed_output_path(prefix: str = "speed") -> Path:
    """Create a unique output path for speed estimation results.

    Args:
        prefix: Filename prefix for the output file.

    Returns:
        Path under the configured output directory.

    Examples:
        >>> path = build_speed_output_path()
        >>> path.name.startswith("speed_")
        True
    """
    return OUTPUT_DIR / f"{prefix}_{uuid4().hex[:8]}.mp4"
