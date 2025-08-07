from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from supervision.video.backends.base import Backend
from supervision.video.common import VideoInfo, Writer


class _OpenCVWriter:
    """Simple wrapper around :class:`cv2.VideoWriter` implementing ``Writer``."""

    def __init__(self, path: str, info: VideoInfo, codec: str) -> None:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(path, fourcc, info.fps, info.resolution_wh)
        if not self._writer.isOpened():
            raise RuntimeError(
                f"Failed to open VideoWriter at {path} using codec {codec}."
            )

    def write(self, frame: np.ndarray) -> None:  # type: ignore[override]
        self._writer.write(frame)

    def close(self) -> None:  # type: ignore[override]
        self._writer.release()


class OpenCVBackend(Backend):
    """Backend that relies on the high-level OpenCV *cv2* API."""

    # The backend keeps no state – all info is inside the *handle* returned by
    # :py:meth:`open` (a :class:`cv2.VideoCapture` instance).

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def open(self, source: Any) -> cv2.VideoCapture:  # type: ignore[override]
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")
        return cap

    def info(self, handle: cv2.VideoCapture) -> VideoInfo:  # type: ignore[override]
        width = int(handle.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(handle.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = handle.get(cv2.CAP_PROP_FPS)
        # Guard against some cameras returning 0 for fps
        fps = int(fps) if fps and fps > 0 else 30
        total = int(handle.get(cv2.CAP_PROP_FRAME_COUNT))
        total = total if total > 0 else None
        return VideoInfo(width=width, height=height, fps=fps, total_frames=total)

    # ------------------------------------------------------------------
    # Reading helpers
    # ------------------------------------------------------------------
    def read(self, handle: cv2.VideoCapture):  # type: ignore[override]
        return handle.read()

    def grab(self, handle: cv2.VideoCapture):  # type: ignore[override]
        return handle.grab()

    def seek(self, handle: cv2.VideoCapture, frame_idx: int):  # type: ignore[override]
        handle.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    # ------------------------------------------------------------------
    # Writing helpers
    # ------------------------------------------------------------------
    def writer(self, path: str, info: VideoInfo, codec: str = "mp4v") -> Writer:  # type: ignore[override]
        return _OpenCVWriter(path=path, info=info, codec=codec)
