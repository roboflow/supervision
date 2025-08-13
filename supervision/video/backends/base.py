from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import numpy as np

from ..utils import VideoInfo


@runtime_checkable
class Backend(Protocol):
    """Protocol for video backends used by ``supervision.video.Video``.

    The high-level adapter instantiates a backend selected by name and only
    calls the methods defined in this protocol. Other members are considered
    private implementation details.
    """

    def __init__(self, source: str | int):
        """Create a new backend for a source.

        Args:
            source: Either a path/URL (``str``) or a webcam index (``int``).
        """

    def info(self) -> VideoInfo:
        """Return static information about the source (size, fps, frames).

        Returns:
            VideoInfo: Static properties of the video source.
        """

    def read(self) -> tuple[bool, np.ndarray]:
        """Decode the next frame.

        Returns:
            tuple[bool, np.ndarray]: ``(success, frame)`` where frame is a
            ``np.ndarray`` (H x W x 3) in backend-specific channel order.
        """

    def grab(self) -> bool:
        """Grab the next frame without decoding pixels.

        Useful to skip frames quickly (for example, when ``stride > 1``).

        Returns:
            bool: ``True`` if a frame position was advanced, ``False`` otherwise.
        """

    def seek(self, frame_idx: int) -> None:
        """Seek so that the next call to ``read`` returns ``frame_idx``."""

    def writer(
        self,
        path: str,
        info: VideoInfo,
        codec: str | None = None,
    ) -> Writer:
        """Return a writer that encodes frames to ``path``.

        Args:
            path: Target file path.
            info: Expected output resolution and fps (copied from source by default).
            codec: FourCC or codec name to override the backend default.

        Returns:
            Writer: A context-manager compatible writer instance.
        """

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield successive frames until exhaustion.

        This is considered convenience behavior; the default implementation is
        sufficient for most backends.

        Yields:
            np.ndarray: The next decoded frame.
        """


@runtime_checkable
class Writer(Protocol):
    """Protocol for encoded video writers returned by ``Backend.writer``."""

    def write(self, frame: np.ndarray, frame_number: int, callback) -> None:
        """Write a single frame to the output stream."""

    def close(self) -> None:
        """Flush and close the underlying container or file descriptor."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False  # propagate exception (if any)


class _NullWriter:
    """Fallback writer that silently drops every frame."""

    def write(self, frame: np.ndarray, frame_number: int, callback) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


__all__ = [
    "Backend",
    "Writer",
]
