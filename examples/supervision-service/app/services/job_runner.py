from pathlib import Path

from app.config import DEFAULT_CONFIDENCE, DEFAULT_IOU, DEFAULT_SPEED_WEIGHTS, DEFAULT_WEIGHTS
from app.db.repository import mark_job_completed, mark_job_failed, update_job_progress
from app.services.speed_processor import build_speed_output_path, estimate_speed_video
from app.services.video_processor import build_output_path, track_video


def _make_progress_callback(job_id: str):
    def callback(current_frame: int, total_frames: int) -> None:
        if total_frames <= 0:
            return
        frame_progress = int(current_frame / total_frames * 95)
        update_job_progress(
            job_id,
            progress=min(frame_progress, 95),
            current_frame=current_frame,
            total_frames=total_frames,
        )

    return callback


def run_track_job(
    job_id: str,
    source_video_path: Path,
    confidence_threshold: float = DEFAULT_CONFIDENCE,
    iou_threshold: float = DEFAULT_IOU,
) -> None:
    """Run tracking in background and update job progress."""
    output_path = build_output_path("tracked")
    try:
        track_video(
            source_video_path=source_video_path,
            target_video_path=output_path,
            weights_path=DEFAULT_WEIGHTS,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            on_progress=_make_progress_callback(job_id),
        )
        update_job_progress(job_id, progress=99, current_frame=0, total_frames=0)
        mark_job_completed(job_id, output_path)
    except Exception as exc:
        mark_job_failed(job_id, str(exc))


def run_speed_job(
    job_id: str,
    source_video_path: Path,
    source_points: list[list[int]],
    target_width: float,
    target_height: float,
    confidence_threshold: float = DEFAULT_CONFIDENCE,
    iou_threshold: float = DEFAULT_IOU,
) -> None:
    """Run speed estimation in background and update job progress."""
    output_path = build_speed_output_path()
    try:
        estimate_speed_video(
            source_video_path=source_video_path,
            target_video_path=output_path,
            source_points=source_points,
            target_width=target_width,
            target_height=target_height,
            weights_path=DEFAULT_SPEED_WEIGHTS,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            on_progress=_make_progress_callback(job_id),
        )
        update_job_progress(job_id, progress=99, current_frame=0, total_frames=0)
        mark_job_completed(job_id, output_path)
    except Exception as exc:
        mark_job_failed(job_id, str(exc))
