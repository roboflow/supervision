from pathlib import Path

import cv2
import numpy as np
import pytest

import supervision as sv


def make_video(
    path: Path, w: int = 160, h: int = 96, fps: int = 20, frames: int = 24
) -> None:
    """Create a small synthetic test video with predictable frame-colors."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert writer.isOpened(), "Failed to open VideoWriter"
    for i in range(frames):
        v = (i * 11) % 250
        frame = np.full((h, w, 3), (v, 255 - v, (2 * v) % 255), np.uint8)
        writer.write(frame)
    writer.release()


def read_frames(path: Path) -> list[np.ndarray]:
    """Read all frames from a video into memory."""
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"Cannot open video: {path}"
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(frame)
    cap.release()
    return out


def frames_equal(a: np.ndarray, b: np.ndarray, max_abs_tol: int = 0) -> bool:
    """Return True if frames are the same within acertain tolerance."""
    if a.shape != b.shape:
        return False
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return diff.max() <= max_abs_tol


def callback_noop(frame: np.ndarray, idx: int) -> np.ndarray:
    """No-op callback: validates pure pipeline correctness."""
    return frame


def callbackb_opencv(frame: np.ndarray, idx: int) -> np.ndarray:
    """
    Simulations some cv2 task...
    """
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize(
    "callback", [callback_noop, callbackb_opencv], ids=["identity", "opencv"]
)
def test_process_video_vs_threads_same_output(callback, tmp_path: Path):
    """
    Ensure that process_video() and process_video_threads() produce identical
    results for the same synthetic source video and callback.
    """
    name = callback.__name__
    src = tmp_path / f"src_{name}.mp4"
    dst_single = tmp_path / f"out_single_{name}.mp4"
    dst_threads = tmp_path / f"out_threads_{name}.mp4"

    make_video(src, frames=24)

    sv.utils.video.process_video(
        source_path=str(src),
        target_path=str(dst_single),
        callback=callback,
        show_progress=False,
    )
    sv.utils.video.process_video_threads(
        source_path=str(src),
        target_path=str(dst_threads),
        callback=callback,
        prefetch=4,
        writer_buffer=4,
        show_progress=False,
    )

    frames_single = read_frames(dst_single)
    frames_threads = read_frames(dst_threads)

    assert len(frames_single) == len(frames_threads) != 0, "Frame count mismatch."

    for i, (fs, ft) in enumerate(zip(frames_single, frames_threads)):
        assert frames_equal(fs, ft), f"Frame {i} is different."
