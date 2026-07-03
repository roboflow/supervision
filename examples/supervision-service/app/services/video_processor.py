from collections.abc import Callable
from pathlib import Path

from ultralytics import YOLO

import supervision as sv

from app.config import DEFAULT_CONFIDENCE, DEFAULT_IOU, DEFAULT_WEIGHTS
from app.services.video_encoding import ensure_browser_playable

ProgressCallback = Callable[[int, int], None]


def track_video(
    source_video_path: Path,
    target_video_path: Path,
    weights_path: Path = DEFAULT_WEIGHTS,
    confidence_threshold: float = DEFAULT_CONFIDENCE,
    iou_threshold: float = DEFAULT_IOU,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Run YOLO detection and ByteTrack on a video file."""
    model = YOLO(str(weights_path))
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    video_info = sv.VideoInfo.from_video_path(str(source_video_path))
    frame_generator = sv.get_video_frames_generator(str(source_video_path))
    total_frames = video_info.total_frames

    if on_progress and total_frames:
        on_progress(0, total_frames)

    target_video_path.parent.mkdir(parents=True, exist_ok=True)

    with sv.VideoSink(str(target_video_path), video_info) as sink:
        for frame_index, frame in enumerate(frame_generator, start=1):
            results = model(
                frame,
                verbose=False,
                conf=confidence_threshold,
                iou=iou_threshold,
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            annotated_frame = box_annotator.annotate(
                scene=frame.copy(),
                detections=detections,
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
            )
            sink.write_frame(frame=annotated_frame)

            if on_progress and total_frames:
                on_progress(frame_index, total_frames)

    if on_progress and total_frames:
        on_progress(total_frames, total_frames)

    ensure_browser_playable(target_video_path)
    return target_video_path


def build_output_path(prefix: str = "tracked") -> Path:
    """Create a unique output video path."""
    from uuid import uuid4

    from app.config import OUTPUT_DIR

    return OUTPUT_DIR / f"{prefix}_{uuid4().hex[:8]}.mp4"
