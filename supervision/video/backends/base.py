from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import numpy as np

from ..utils import VideoInfo


@runtime_checkable
class Backend(Protocol):
    """
    The high-level :pyclass:`~supervision.video.Video` adapter instantiates a
    backend - selected by name - and then only calls the methods defined
    below.  Anything else is considered a private implementation detail.
    """

    def __init__(self, source: str | int):
        """Create a new backend for source.

        ``source`` can be
        * ``str`` - file path, RTSP/HTTP URL …
        * ``int`` - webcam index (OpenCV-style)
        """

    def info(self) -> VideoInfo:
        """Return static information (width / height / fps / total_frames)."""

    def read(self) -> tuple[bool, np.ndarray]:
        """Decode the next frame.

        Returns ``(success, frame)`` where frame is a ``np.ndarray`` (HxWx3).
        """

    def grab(self) -> bool:
        """Grab the next frame without decoding pixels.

        Equivalent to OpenCV's ``VideoCapture.grab``.  Useful if the user only
        wants to skip frames quickly (stride > 1 for example).
        """

    def seek(self, frame_idx: int) -> None:
        """Seek to frame_idx so that the next :py:meth:`read` returns it."""

    # Encoding ---------------------------------------------------------------

    def writer(
        self,
        path: str,
        info: VideoInfo,
        codec: str | None = None,
    ) -> Writer:
        """Return a writer that encodes frames to path.

        Parameters
        ----------
        path:
            Target file path.
        info:
            Expected output resolution / fps (copied from source by default).
        codec:
            FourCC / codec name to override the backend default.
        """

    # Iterator convenience ---------------------------------------------------

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield successive frames until exhaustion.

        This is considered convenience behaviour; the default implementation
        below is fine for most back-ends.
        """


@runtime_checkable
class Writer(Protocol):
    """Protocol for an encoded video writer returned by :py:meth:`Backend.writer`."""

    def write(self, frame: np.ndarray, frame_number: int, callback) -> None:
        """Write a single BGR / RGB frame to the output stream."""

    def close(self) -> None:
        """Flush and close the underlying container / file descriptor."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False  # propagate exception (if any)


# ---------------------------------------------------------------------------
# Utility - a dummy writer that does nothing.  Useful for testing.
# ---------------------------------------------------------------------------


class _NullWriter:
    """Fallback Writer that silently drops every frame."""

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
