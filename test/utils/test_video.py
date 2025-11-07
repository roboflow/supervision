import tempfile

import cv2
import numpy as np

from supervision.utils.video import process_video


def create_test_video(path, num_frames, width=20, height=10):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 1.0, (width, height))

    for _ in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        out.write(frame)

    out.release()


def test_process_video_max_frames_exceeds_total_frames():
    with (
        tempfile.NamedTemporaryFile(suffix=".mp4") as source_file,
        tempfile.NamedTemporaryFile(suffix=".mp4") as target_file,
    ):
        create_test_video(source_file.name, num_frames=5)

        process_video(
            source_path=source_file.name,
            target_path=target_file.name,
            callback=lambda frame, _: frame,
            max_frames=10,
        )
