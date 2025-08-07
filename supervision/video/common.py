from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class VideoInfo:
    """Container for video metadata.

    Attributes
    ----------
    width:
        Width of the video in pixels.
    height:
        Height of the video in pixels.
    fps:
        Frames-per-second of the stream.
    total_frames:
        Optional number of frames in the stream. For live streams or webcams this
        value can be *None*.
    """

    width: int
    height: int
    fps: int
    total_frames: int | None = None

    # Convenience helpers --------------------------------------------------
    @property
    def resolution_wh(self) -> tuple[int, int]:
        """Return width/height tuple matching cv2 / numpy order."""

        return self.width, self.height


class Writer(Protocol):
    """Protocol describing minimal video writer interface used by ``Video``.

    Any backend-specific writer must implement these two methods.
    """

    def write(self, frame: np.ndarray) -> None:  # pragma: no cover
        ...

    def close(self) -> None:  # pragma: no cover
        ...

