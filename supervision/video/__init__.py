from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from collections.abc import Callable

import cv2  # Only needed for resize utility even when using PyAV backend
import numpy as np
from tqdm.auto import tqdm

from supervision.utils.internal import warn_deprecated
from supervision.video.backends.base import Backend
from supervision.video.backends.opencv_backend import OpenCVBackend
from supervision.video.backends.pyav_backend import PyAVBackend
from supervision.video.common import VideoInfo, Writer

logger = logging.getLogger(__name__)

# Public API -----------------------------------------------------------------
__all__ = [
    "Video",
    "VideoInfo",
]


_BACKENDS: dict[str, Backend] = {
    "opencv": OpenCVBackend(),
    "pyav": PyAVBackend(),
}


def _get_backend(name: str) -> Backend:
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown video backend '{name}'. Supported backends: {list(_BACKENDS)}"
        )
    return _BACKENDS[name]


class _WriterContext:
    """Context manager around backend *Writer*."""

    def __init__(self, writer: Writer):
        self._writer = writer

    def __enter__(self):
        return self

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._writer.close()


class Video:
    """Unified abstraction for reading and writing videos.

    The class acts as a thin wrapper delegating heavy-lifting to backend
    implementations (OpenCV by default). It provides convenient helpers for
    iterating over frames, slicing, resizing on-the-fly and saving processed
    videos.
    """

    def __init__(self, source: Any, *, backend: str = "opencv") -> None:
        self._backend_name = backend.lower()
        self._backend: Backend = _get_backend(self._backend_name)
        self._handle = self._backend.open(source)
        self._info = self._backend.info(self._handle)
        self._source = source  # keep for reference

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def info(self) -> VideoInfo:
        """Return :class:`VideoInfo` describing the stream."""

        return self._info

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over **all** frames in the stream."""

        while True:
            success, frame = self._backend.read(self._handle)
            if not success:
                break
            yield frame

    def frames(
        self,
        *,
        stride: int = 1,
        start: int = 0,
        end: int | None = None,
        resolution_wh: tuple[int, int] | None = None,
    ) -> Iterator[np.ndarray]:
        """Advanced frame generator.

        Parameters
        ----------
        stride:
            Return every *stride*-th frame. *1* returns consecutive frames.
        start:
            Frame index to start reading from (inclusive).
        end:
            Frame index to stop before (exclusive). ``None`` means *read to end*.
        resolution_wh:
            Optionally resize every returned frame to this *(width, height)* while
            maintaining **BGR** channel order.
        """

        if start:
            try:
                self._backend.seek(self._handle, start)
            except Exception as e:
                logger.debug(
                    "Backend seek failed (%s). Falling back to iterative seek.", e
                )
                # fallback: iterative grabbing
                for _ in range(start):
                    ok = self._backend.grab(self._handle)
                    if not ok:
                        return  # out-of-stream

        idx = start
        while True:
            success, frame = self._backend.read(self._handle)
            if not success:
                break
            if end is not None and idx >= end:
                break

            if resolution_wh is not None:
                frame = cv2.resize(frame, resolution_wh)
            yield frame

            # skip *stride-1* frames via grab to reduce decoding cost
            for _ in range(stride - 1):
                idx += 1
                if end is not None and idx >= end:
                    break
                ok = self._backend.grab(self._handle)
                if not ok:
                    break
            idx += 1

    # ------------------------------------------------------------------
    # Saving helpers
    # ------------------------------------------------------------------
    def save(
        self,
        target_path: str,
        *,
        callback: Callable[[np.ndarray, int], np.ndarray],
        show_progress: bool = False,
        fps: int | None = None,
        width: int | None = None,
        height: int | None = None,
        codec: str = "mp4v",
    ) -> None:
        """Process the stream frame-by-frame and save as a new video.

        The *callback* is executed **synchronously** for every frame and must
        return a processed frame with the **same resolution** as provided to the
        writer.
        """

        # Determine target info -------------------------------------------------
        target_info = VideoInfo(
            width=width or self._info.width,
            height=height or self._info.height,
            fps=fps or self._info.fps,
        )

        writer = self._backend.writer(target_path, info=target_info, codec=codec)

        frame_iter = self.frames()
        if show_progress:
            total = self._info.total_frames
            frame_iter = tqdm(frame_iter, total=total, desc="Saving video")

        for i, frame in enumerate(frame_iter):
            processed = callback(frame, i)
            writer.write(processed)
        writer.close()

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------
    def sink(
        self, target_path: str, *, info: VideoInfo | None = None, codec: str = "mp4v"
    ) -> _WriterContext:
        """Return a context manager for manually writing frames."""

        return _WriterContext(
            self._backend.writer(target_path, info or self._info, codec=codec)
        )

    # ------------------------------------------------------------------
    # Cleanup helpers
    # ------------------------------------------------------------------
    def release(self):
        """Release underlying handle (if applicable)."""

        # Only OpenCV backend currently needs explicit release
        release_func = getattr(self._handle, "release", None)
        if callable(release_func):
            release_func()

    def __del__(self):
        self.release()

    # ------------------------------------------------------------------
    # Legacy helpers bridging old API
    # ------------------------------------------------------------------
    @staticmethod
    def _warn_deprecated_old_api(name: str):
        warn_deprecated(
            f"{name} will be removed in a future release. Use the new supervision.Video class instead."
        )
