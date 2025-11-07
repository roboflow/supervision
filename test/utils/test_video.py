import cv2
import numpy as np
import pytest

from supervision.utils.video import process_video


def create_test_video(path, num_frames, width=20, height=10):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 1.0, (width, height))

    for _ in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        out.write(frame)

    out.release()


def test_process_video_max_frames_exceeds_total_frames(tmp_path):
    source_path = tmp_path / "source.mp4"
    target_path = tmp_path / "target.mp4"

    create_test_video(str(source_path), num_frames=5)

    process_video(
        source_path=str(source_path),
        target_path=str(target_path),
        callback=lambda frame, _: frame,
        max_frames=10,
    )
