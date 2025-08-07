from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generator, Iterable, Optional, Protocol, Tuple, Union

import cv2
import numpy as np

from supervision.utils.internal import warn_deprecated


# -----------------------------
# Public data structures
# -----------------------------


@dataclass
class VideoInfo:
    """
    Describes basic video properties.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second. For streams this may be a non-integer (e.g., 29.97) or
            0.0 when unknown. Prefer using the backend/container value without
            rounding to maintain accuracy (see issue #1687).
        total_frames: Total frames if known, otherwise None (e.g., live streams).
    """

    width: int
    height: int
    fps: float
    total_frames: Optional[int] = None

    @property
    def resolution_wh(self) -> Tuple[int, int]:
        return self.width, self.height


# -----------------------------
# Backend protocol
# -----------------------------


class Writer(Protocol):
    def write(self, frame: np.ndarray) -> None: ...

    def close(self) -> None: ...


class Backend(Protocol):
    def open(self, source: Union[str, int]) -> Any: ...

    def info(self, handle: Any) -> VideoInfo: ...

    def read(self, handle: Any) -> Tuple[bool, Optional[np.ndarray]]: ...

    def grab(self, handle: Any) -> bool: ...

    def seek(self, handle: Any, frame_idx: int) -> None: ...

    def writer(self, path: str, info: VideoInfo, codec: str) -> Writer: ...


# -----------------------------
# OpenCV backend
# -----------------------------


class _OpenCVWriter:
    def __init__(self, path: str, info: VideoInfo, codec: str):
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
        except TypeError:
            # Fallback for safety
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, info.fps if info.fps > 0 else 30.0, info.resolution_wh)

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()


class OpenCVBackend:
    def open(self, source: Union[str, int]) -> Any:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        return cap

    def info(self, handle: Any) -> VideoInfo:
        width = int(handle.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(handle.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(handle.get(cv2.CAP_PROP_FPS) or 0.0)
        # For some streams OpenCV reports 0 or NaN. Normalize to 0.0 as unknown.
        fps = fps if np.isfinite(fps) and fps > 0 else 0.0
        total = int(handle.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(total) or total <= 0:
            total = None
        return VideoInfo(width=width, height=height, fps=fps, total_frames=total)

    def read(self, handle: Any) -> Tuple[bool, Optional[np.ndarray]]:
        ok, frame = handle.read()
        if not ok:
            return False, None
        return True, frame

    def grab(self, handle: Any) -> bool:
        return handle.grab()

    def seek(self, handle: Any, frame_idx: int) -> None:
        handle.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def writer(self, path: str, info: VideoInfo, codec: str) -> Writer:
        return _OpenCVWriter(path, info, codec)


# -----------------------------
# PyAV backend (optional)
# -----------------------------


class _PyAVWriter:
    def __init__(self, path: str, info: VideoInfo, codec: str):
        try:
            import av  # type: ignore
        except Exception as e:  # pragma: no cover - import error path
            raise RuntimeError("PyAV is required for 'pyav' backend. Install 'av'.") from e

        self._av = av
        self._container = av.open(path, mode="w")
        stream = self._container.add_stream(codec or "libx264", rate=info.fps if info.fps > 0 else 30.0)
        stream.width = info.width
        stream.height = info.height
        stream.pix_fmt = "yuv420p"
        self._stream = stream

    def write(self, frame: np.ndarray) -> None:
        av = self._av
        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        for packet in self._stream.encode(video_frame):
            self._container.mux(packet)

    def close(self) -> None:
        # Flush
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()


class PyAVBackend:
    def __init__(self) -> None:
        try:
            import av  # type: ignore
        except Exception as e:  # pragma: no cover - import error path
            raise RuntimeError("PyAV is required for 'pyav' backend. Install 'av'.") from e
        self._av = av

    def open(self, source: Union[str, int]) -> Any:
        if isinstance(source, int):
            # PyAV does not handle webcams by index directly; use OpenCV instead.
            raise RuntimeError("PyAV backend does not support webcam index sources. Use 'opencv' backend.")
        try:
            container = self._av.open(source)
        except Exception as e:
            raise RuntimeError(f"Could not open video source with PyAV: {source}") from e
        return container

    def info(self, handle: Any) -> VideoInfo:
        # Prefer container-level metadata for precise fps
        streams = [s for s in handle.streams if s.type == "video"]
        if not streams:
            # Fallback to zeros
            return VideoInfo(width=0, height=0, fps=0.0, total_frames=None)
        vs = streams[0]
        width = int(vs.width or 0)
        height = int(vs.height or 0)
        # fps may be a Fraction
        if getattr(vs, "average_rate", None):
            try:
                fps = float(vs.average_rate)
            except Exception:
                fps = 0.0
        elif getattr(vs, "rate", None):
            try:
                fps = float(vs.rate)
            except Exception:
                fps = 0.0
        else:
            fps = 0.0
        # total_frames can be derived for files; for some codecs it's None
        total = getattr(vs, "frames", None)
        total_frames = int(total) if isinstance(total, (int, np.integer)) and total > 0 else None
        return VideoInfo(width=width, height=height, fps=fps if fps > 0 else 0.0, total_frames=total_frames)

    def read(self, handle: Any) -> Tuple[bool, Optional[np.ndarray]]:
        av = self._av
        for packet in handle.demux(video=0):
            for frame in packet.decode():
                img = frame.to_ndarray(format="bgr24")
                return True, img
        return False, None

    def grab(self, handle: Any) -> bool:
        # PyAV does not support lightweight grab; perform a read and drop
        ok, _ = self.read(handle)
        return ok

    def seek(self, handle: Any, frame_idx: int) -> None:
        # Seek by timestamp when possible. We approximate using fps if known.
        info = self.info(handle)
        if info.fps > 0:
            ts = int((frame_idx / info.fps) * 1e6)  # microseconds
            handle.seek(ts, any_frame=False, backward=True)
        else:
            # If fps unknown, seek to start as safe default
            handle.seek(0)

    def writer(self, path: str, info: VideoInfo, codec: str) -> Writer:
        return _PyAVWriter(path, info, codec)


# -----------------------------
# Public Video API
# -----------------------------


_BACKENDS = {
    "opencv": OpenCVBackend,
    "pyav": PyAVBackend,
}


class Video(Iterable[np.ndarray]):
    """
    Unified video interface with multi-backend support.

    Examples:
        video = Video("source.mp4").info
        for frame in Video("source.mp4"): ...
        for frame in Video("source.mp4").frames(stride=5, start=100, end=500, resolution_wh=(1280, 720)): ...

        def cb(frame, i): return frame
        Video("source.mp4").save("out.mp4", callback=cb, show_progress=True)
    """

    def __init__(
        self,
        source: Union[str, int],
        backend: Union[str, Backend] = "opencv",
    ) -> None:
        if isinstance(backend, str):
            backend_lower = backend.lower()
            if backend_lower not in _BACKENDS:
                raise ValueError(f"Unknown backend '{backend}'. Supported: {list(_BACKENDS)}")
            self._backend: Backend = _BACKENDS[backend_lower]()
            self._backend_name = backend_lower
        else:
            self._backend = backend
            self._backend_name = backend.__class__.__name__

        self._source = source
        self._handle = self._backend.open(source)
        self._cached_info: Optional[VideoInfo] = None

    @property
    def info(self) -> VideoInfo:
        if self._cached_info is None:
            self._cached_info = self._backend.info(self._handle)
            # If fps is unknown for live streams, prefer 0.0 instead of rounding
            if self._cached_info.fps is None:
                self._cached_info.fps = 0.0  # type: ignore[assignment]
        return self._cached_info

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        yield from self.frames()

    def frames(
        self,
        stride: int = 1,
        start: int = 0,
        end: Optional[int] = None,
        resolution_wh: Optional[Tuple[int, int]] = None,
        iterative_seek: bool = False,
    ) -> Generator[np.ndarray, None, None]:
        """Iterate frames with optional stride, range and on-the-fly resize."""
        # Setup start/end
        start = max(start, 0)
        info = self.info
        total = info.total_frames if info.total_frames is not None else end
        if end is None and total is not None:
            end = total

        # Seek to start
        if iterative_seek and start > 0:
            for _ in range(start):
                if not self._backend.grab(self._handle):
                    return
        elif start > 0:
            try:
                self._backend.seek(self._handle, start)
            except Exception:
                # If backend cannot seek, fallback to grabbing
                for _ in range(start):
                    if not self._backend.grab(self._handle):
                        return

        frame_position = start
        while True:
            ok, frame = self._backend.read(self._handle)
            if not ok or frame is None:
                break
            if end is not None and frame_position >= end:
                break

            if resolution_wh is not None and (frame.shape[1], frame.shape[0]) != resolution_wh:
                frame = cv2.resize(frame, resolution_wh)

            yield frame

            # Skip stride-1 frames
            for _ in range(stride - 1):
                if not self._backend.grab(self._handle):
                    return
                frame_position += 1
            frame_position += 1

    def save(
        self,
        target_path: str,
        callback,
        *,
        fps: Optional[float] = None,
        codec: str = "mp4v",
        show_progress: bool = False,
        progress_message: str = "Processing video",
        stride: int = 1,
        start: int = 0,
        end: Optional[int] = None,
        resolution_wh: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Process and save to a new video using a callback(frame, index) -> frame."""
        from tqdm.auto import tqdm  # lightweight import

        src_info = self.info
        target_info = VideoInfo(
            width=resolution_wh[0] if resolution_wh else src_info.width,
            height=resolution_wh[1] if resolution_wh else src_info.height,
            fps=float(fps if fps is not None else (src_info.fps if src_info.fps > 0 else 30.0)),
            total_frames=None,
        )

        writer = self._backend.writer(target_path, target_info, codec)

        # total for progress bar if known
        total_frames: Optional[int] = None
        if end is not None:
            total_frames = max(0, (end - start + (stride - 1)) // stride)
        elif src_info.total_frames is not None:
            total_frames = max(0, (src_info.total_frames - start + (stride - 1)) // stride)

        try:
            iterator: Iterable[np.ndarray] = self.frames(
                stride=stride, start=start, end=end, resolution_wh=resolution_wh
            )
            if show_progress:
                iterator = tqdm(iterator, total=total_frames, desc=progress_message, disable=not show_progress)
            for idx, frame in enumerate(iterator):
                out = callback(frame, idx)
                writer.write(out)
        finally:
            writer.close()

    def sink(
        self, target_path: str, *, info: Optional[VideoInfo] = None, codec: str = "mp4v", fps: Optional[float] = None
    ) -> Writer:
        """Create a writer for manual control.

        If 'info' is not provided, it is derived from the source. You can override
        FPS via the 'fps' parameter.
        """
        src_info = self.info
        if info is None:
            info = VideoInfo(
                width=src_info.width,
                height=src_info.height,
                fps=float(fps if fps is not None else (src_info.fps if src_info.fps > 0 else 30.0)),
                total_frames=None,
            )
        else:
            if fps is not None:
                warn_deprecated("'fps' parameter is ignored when 'info' is provided. Set 'info.fps' instead.")
        return self._backend.writer(target_path, info, codec)


