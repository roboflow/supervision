from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any, Optional, Protocol, Tuple, Union
from collections.abc import Callable

import numpy as np

# Try to import OpenCV and PyAV
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import av
except ImportError:
    av = None

from supervision.utils.video import VideoInfo  # Reuse or extend as needed


class Backend(Protocol):
    def open(self, source: str | int) -> Any: ...
    def info(self, handle: Any) -> VideoInfo: ...
    def read(self, handle: Any) -> tuple[bool, np.ndarray]: ...
    def grab(self, handle: Any) -> bool: ...
    def seek(self, handle: Any, frame_idx: int) -> None: ...
    def writer(self, path: str, info: VideoInfo, codec: str) -> Writer: ...


class Writer(Protocol):
    def write(self, frame: np.ndarray) -> None: ...
    def close(self) -> None: ...


class OpenCVBackend:
    def open(self, source: str | int) -> Any:
        if cv2 is None:
            raise ImportError("OpenCV is not installed.")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise Exception(f"Could not open video source: {source}")
        return cap

    def info(self, handle: Any) -> VideoInfo:
        width = int(handle.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(handle.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = handle.get(cv2.CAP_PROP_FPS)
        total_frames = int(handle.get(cv2.CAP_PROP_FRAME_COUNT))
        return VideoInfo(width, height, fps, total_frames)

    def read(self, handle: Any) -> tuple[bool, np.ndarray]:
        return handle.read()

    def grab(self, handle: Any) -> bool:
        return handle.grab()

    def seek(self, handle: Any, frame_idx: int) -> None:
        handle.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def writer(self, path: str, info: VideoInfo, codec: str) -> Writer:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, info.fps, (info.width, info.height))
        return OpenCVWriter(writer)


class OpenCVWriter:
    def __init__(self, writer):
        self.writer = writer

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()


class PyAVBackend:
    def open(self, source: str | int) -> Any:
        if av is None:
            raise ImportError("PyAV is not installed.")
        return av.open(source)

    def info(self, handle: Any) -> VideoInfo:
        # TODO: Implement PyAV info extraction
        raise NotImplementedError

    def read(self, handle: Any) -> tuple[bool, np.ndarray]:
        # TODO: Implement PyAV frame reading
        raise NotImplementedError

    def grab(self, handle: Any) -> bool:
        # TODO: Implement PyAV frame grabbing
        raise NotImplementedError

    def seek(self, handle: Any, frame_idx: int) -> None:
        # TODO: Implement PyAV seeking
        raise NotImplementedError

    def writer(self, path: str, info: VideoInfo, codec: str) -> Writer:
        # TODO: Implement PyAV writer
        raise NotImplementedError


class Video:
    def __init__(self, source: str | int, backend: str = "opencv"):
        self.source = source
        self.backend_name = backend
        self.backend = self._select_backend(backend)
        self.handle = self.backend.open(source)
        self._info = self.backend.info(self.handle)
        logging.info(f"Video opened: {source} with backend {backend}")
        logging.info(f"Video info: {self._info}")

    @staticmethod
    def _select_backend(name: str) -> Backend:
        if name == "opencv":
            return OpenCVBackend()
        elif name == "pyav":
            return PyAVBackend()
        else:
            raise ValueError(f"Unknown backend: {name}")

    @property
    def info(self) -> VideoInfo:
        return self._info

    def __iter__(self):
        return self.frames()

    def frames(
        self,
        stride: int = 1,
        start: int = 0,
        end: int | None = None,
        resolution_wh: tuple[int, int] | None = None,
    ) -> Generator[np.ndarray]:
        self.backend.seek(self.handle, start)
        frame_idx = start
        while True:
            success, frame = self.backend.read(self.handle)
            if not success or (end is not None and frame_idx >= end):
                break
            if resolution_wh is not None:
                frame = self._resize(frame, resolution_wh)
            yield frame
            for _ in range(stride - 1):
                if not self.backend.grab(self.handle):
                    break
                frame_idx += 1
            frame_idx += 1

    def _resize(self, frame: np.ndarray, resolution_wh: tuple[int, int]) -> np.ndarray:
        if cv2 is None:
            raise ImportError("OpenCV is required for resizing.")
        return cv2.resize(frame, resolution_wh)

    def save(
        self,
        target_path: str,
        callback: Callable[[np.ndarray, int], np.ndarray] | None = None,
        show_progress: bool = False,
        fps: float | None = None,
        resolution_wh: tuple[int, int] | None = None,
        codec: str = "mp4v",
    ) -> None:
        info = self.info
        if fps is not None:
            info = VideoInfo(info.width, info.height, fps, info.total_frames)
        if resolution_wh is not None:
            info = VideoInfo(
                resolution_wh[0], resolution_wh[1], info.fps, info.total_frames
            )
        writer = self.backend.writer(target_path, info, codec)
        try:
            for idx, frame in enumerate(self.frames(resolution_wh=resolution_wh)):
                if callback:
                    frame = callback(frame, idx)
                writer.write(frame)
        finally:
            writer.close()

    def sink(
        self, target_path: str, info: VideoInfo | None = None, codec: str = "mp4v"
    ) -> Writer:
        if info is None:
            info = self.info
        return self.backend.writer(target_path, info, codec)

    def __del__(self):
        # Release resources
        if hasattr(self.handle, "release"):
            self.handle.release()
