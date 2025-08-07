from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from supervision.video.common import VideoInfo, Writer
from supervision.video.backends.base import Backend

# PyAV is an optional dependency – we import lazily and fail with clear message.
try:
    import av  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    av = None  # type: ignore


class _PyAVWriter(Writer):
    """Simple PyAV writer wrapping a mux and encoded stream."""

    def __init__(self, path: str, info: VideoInfo, codec: str):
        if av is None:
            raise RuntimeError("PyAV is not installed. Install with `pip install av`." )
        self._container = av.open(path, mode="w")
        self._stream = self._container.add_stream(codec, rate=info.fps)
        self._stream.width = info.width
        self._stream.height = info.height
        self._stream.pix_fmt = "yuv420p"

    def write(self, frame: np.ndarray) -> None:  # type: ignore[override]
        frame_rgb = frame[:, :, ::-1]  # BGR to RGB
        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        for packet in self._stream.encode(av_frame):
            self._container.mux(packet)

    def close(self) -> None:  # type: ignore[override]
        # flush encoder
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()


class PyAVBackend(Backend):
    """Backend powered by *PyAV* (ffmpeg bindings) for robust decoding/encoding."""

    def _require_av(self):
        if av is None:
            raise RuntimeError("PyAV backend requested but PyAV is not installed.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def open(self, source: Any):  # type: ignore[override]
        self._require_av()
        return av.open(source)

    def info(self, handle):  # type: ignore[override]
        self._require_av()
        # Assume first video stream
        stream = next((s for s in handle.streams if s.type == "video"), None)
        if stream is None:
            raise RuntimeError("No video stream found.")
        width = stream.codec_context.width
        height = stream.codec_context.height
        fps = int(stream.average_rate) if stream.average_rate else 30
        total_frames = stream.frames if stream.frames else None
        return VideoInfo(width=width, height=height, fps=fps, total_frames=total_frames)

    # Reading -------------------------------------------------------------
    def _frame_iter(self, handle) -> Iterator[np.ndarray]:
        for packet in handle.demux(video=0):
            for frame in packet.decode():
                img = frame.to_ndarray(format="bgr24")
                yield img

    def read(self, handle):  # type: ignore[override]
        try:
            img = next(self._frame_iter(handle))
            return True, img
        except StopIteration:
            return False, None  # type: ignore[return-value]

    def grab(self, handle):  # type: ignore[override]
        # Grabbing without decoding not trivial – decode and drop one frame.
        success, _ = self.read(handle)
        return success

    def seek(self, handle, frame_idx: int):  # type: ignore[override]
        self._require_av()
        # Use timestamp seek – approximate
        stream = next((s for s in handle.streams if s.type == "video"), None)
        if stream is None:
            raise RuntimeError("No video stream found.")
        time_base = stream.time_base
        target_ts = int(frame_idx / stream.average_rate / time_base)
        handle.seek(target_ts, stream=stream)

    # Writing -------------------------------------------------------------
    def writer(self, path: str, info: VideoInfo, codec: str = "libx264") -> Writer:  # type: ignore[override]
        self._require_av()
        return _PyAVWriter(path, info, codec)

