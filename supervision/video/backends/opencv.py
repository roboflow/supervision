from collections.abc import Callable, Iterator
from typing import Any

import cv2
import numpy as np

from ..utils import VideoInfo
from .base import Writer


class OpenCVWriter:
    def __init__(self, vw: cv2.VideoWriter, info: VideoInfo):
        """OpenCV-based video writer.

        Args:
            vw: An initialized ``cv2.VideoWriter`` instance.
            info: Output video information used to validate/resize frames.
        """
        self._vw = vw
        self.info = info

    def write(
        self,
        frame: np.ndarray,
        frame_number: int,
        callback: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        """Write a frame, applying an optional callback and resize if needed.

        Args:
            frame: Input frame array.
            frame_number: Sequential frame number being written.
            callback: Optional function ``(frame, frame_number) -> frame`` to
                transform the frame before writing.
        """
        if callback:
            frame = callback(frame, frame_number)
        if frame.shape[0] != self.info.height or frame.shape[1] != self.info.width:
            frame = cv2.resize(frame, (self.info.width, self.info.height))
        self._vw.write(frame)

    def close(self) -> None:
        """Release the underlying ``cv2.VideoWriter``."""
        self._vw.release()

    def __enter__(self) -> Writer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class OpenCVBackend:
    def __init__(self, source_path: str | int):
        """Create a new backend for a source path or webcam index.

        Args:
            source_path: File path or stream URL (``str``) or webcam index
                (``int``).
        """
        self.source_path = source_path
        self.cap = cv2.VideoCapture(self.source_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source {self.source_path}")

    def info(self) -> VideoInfo:
        """Return static information (width, height, fps, total_frames)."""
        from ..core import VideoInfo

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        precise_fps = self.cap.get(cv2.CAP_PROP_FPS)
        fps = int(round(precise_fps, 0))
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return VideoInfo(w, h, fps, precise_fps, n)

    def read(self) -> tuple[bool, np.ndarray]:
        """Decode the next frame."""
        return self.cap.read()

    def grab(self) -> bool:
        """Grab the next frame without decoding pixels."""
        return self.cap.grab()

    def seek(self, frame_idx: int) -> None:
        """Seek so that the next call to ``read`` returns ``frame_idx``."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    # ? Do we want to mix and match different writers to different backends?
    def writer(self, path: str, info: VideoInfo, codec: str | None = None) -> Writer:
        """Return a writer that encodes frames to a file path.

        Args:
            path: Target file path.
            info: Expected output resolution and fps (copied from source by default).
            codec: FourCC or codec name to override the backend default.
        """
        fourcc = (
            cv2.VideoWriter_fourcc(*codec) if codec else cv2.VideoWriter_fourcc(*"mp4v")
        )
        vw = cv2.VideoWriter(path, fourcc, info.fps, (info.width, info.height))
        return OpenCVWriter(vw, info)

    def frames(
        self,
        stride: int = 1,
        start: int = 0,
        end: int | None = None,
        resolution_wh: tuple[int, int] | None = None,
        interpolation=cv2.INTER_LINEAR,
    ) -> Iterator[np.ndarray]:
        """Yield frames lazily, with optional skipping and resizing.

        Args:
            stride: Number of frames to skip between yielded frames (``1`` yields every frame).
            start: First frame index (0-based) to yield.
            end: Index after the last frame to yield. ``None`` means until exhaustion.
            resolution_wh: Optional ``(width, height)`` to resize each yielded frame to.
            interpolation: OpenCV interpolation flag used when resizing.

        Yields:
            np.ndarray: The next decoded (and optionally resized) video frame.
        """
        if stride < 1:
            raise ValueError("stride must be >= 1")

        info = self.info()
        total = (
            info.total_frames if info.total_frames and info.total_frames > 0 else None
        )
        if end is None and total is not None:
            end = total
        if start < 0 or start >= end:
            return

        # Position capture at the start frame
        self.seek(start)
        current_idx = start
        infinate_stream = end is None

        while infinate_stream or current_idx < end:
            success, frame = self.read()
            if not success:
                break

            if resolution_wh is not None and (
                frame.shape[1] != resolution_wh[0] or frame.shape[0] != resolution_wh[1]
            ):
                frame = cv2.resize(frame, resolution_wh, interpolation=interpolation)

            yield frame
            current_idx += 1

            # Efficiently skip stride-1 frames with grab()
            skip = stride - 1
            while skip and current_idx < end:
                grabbed = self.grab()
                if not grabbed:
                    return
                current_idx += 1
                skip -= 1

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield successive frames until exhaustion.

        This is considered convenience behavior; the default implementation is
        sufficient for most backends.
        """
        while True:
            success, frame = self.read()
            if not success:
                break
            yield frame

    def release(self):
        """Release the video file."""
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()

    def __len__(self) -> int:
        n = self.info().total_frames
        if n is None or n < 0:
            raise TypeError("length is unknown for this stream")
        return n

    def __getitem__(self, index: int) -> np.ndarray:
        current = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = self.read()
        self.cap.set(
            cv2.CAP_PROP_POS_FRAMES, current
        )  # ? Do we want to restore the video to the original position?
        if not success:
            raise IndexError(f"Failed to read frame {index}")
        return frame


# Provide a consistent alias for the core loader
Backend = OpenCVBackend
