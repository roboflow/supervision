from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from supervision.video.common import VideoInfo, Writer


class Backend(Protocol):
    """Protocol that every decoding/encoding backend must follow."""

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------
    def open(self, source: Any):  # pragma: no cover
        """Open the *source* and return an opaque handle.

        The *source* can be anything accepted by the backend (file path, RTSP
        url, integer camera index, etc.). The returned *handle* MUST be passed
        to any other backend method.
        """

        ...

    def info(self, handle: Any) -> VideoInfo:  # pragma: no cover
        """Return :class:`~supervision.video.common.VideoInfo` for the opened stream."""

        ...

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def read(self, handle: Any) -> tuple[bool, np.ndarray]:  # pragma: no cover
        """Return ``(success, frame)`` where *frame* is a **BGR** image."""

        ...

    def grab(self, handle: Any) -> bool:  # pragma: no cover
        """Skip a single frame. Return *False* if grabbing failed/end-of-stream."""

        ...

    def seek(self, handle: Any, frame_idx: int) -> None:  # pragma: no cover
        """Seek to *frame_idx* from the start of the stream."""

        ...

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def writer(
        self, path: str, info: VideoInfo, codec: str = "mp4v"
    ) -> Writer:  # pragma: no cover
        """Return a :class:`Writer` instance for saving frames."""

        ...
