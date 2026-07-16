"""Private PyAV-backed video and audio fallbacks."""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import numpy.typing as npt

from supervision._cv2._common import BackendUnavailableError
from supervision._cv2.constants import (
    _CAP_PROP_FPS,
    _CAP_PROP_FRAME_COUNT,
    _CAP_PROP_FRAME_HEIGHT,
    _CAP_PROP_FRAME_WIDTH,
    _CAP_PROP_POS_FRAMES,
)

logger = logging.getLogger(__name__)

_CODECS = {
    "mp4v": ("mpeg4", "yuv420p"),
    "xvid": ("mpeg4", "yuv420p"),
    "avc1": ("libx264", "yuv420p"),
    "h264": ("libx264", "yuv420p"),
    "mjpg": ("mjpeg", "yuvj420p"),
    "vp09": ("libvpx-vp9", "yuv420p"),
}


def _video_writer_fourcc(*chars: str) -> int:
    """Encode four single-character strings using OpenCV's integer layout."""
    if len(chars) != 4 or any(len(char) != 1 for char in chars):
        raise TypeError("VideoWriter_fourcc requires exactly four characters")
    return sum(ord(char) << (8 * index) for index, char in enumerate(chars))


def _decode_fourcc(fourcc: int) -> str:
    """Decode a fourcc integer into its four-character representation."""
    return "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4))


def _codec_details(fourcc: int) -> tuple[str, str]:
    """Return the PyAV codec and pixel format for a supported fourcc."""
    code = _decode_fourcc(fourcc).lower()
    try:
        return _CODECS[code]
    except KeyError as exc:
        raise ValueError(f"Unsupported video codec: {code!r}") from exc


class _VideoCapture:
    """Expose OpenCV-shaped file capture backed by PyAV decoding."""

    def __init__(self, source: str | os.PathLike[str] | int) -> None:
        """Open a file source and retain a lazy PyAV frame iterator."""
        self._container: Any = None
        self._stream: Any = None
        self._frames: Iterator[Any] | None = None
        self._source = source
        self._position = 0
        self._frame_count_cache: int | None = None
        self._opened = False
        self._error: Exception | None = None

        try:
            if isinstance(source, int):
                raise BackendUnavailableError(
                    "PyAV fallback supports file paths, not webcam device indexes."
                )
            self._container = av.open(str(source), mode="r")
            if not self._container.streams.video:
                raise ValueError(f"Video source has no video stream: {source}")
            self._stream = self._container.streams.video[0]
            self._frames = iter(self._container.decode(video=self._stream.index))
            self._opened = True
        except Exception as exc:
            self._error = exc
            self.release()

    def isOpened(self) -> bool:
        """Return whether the underlying video file is open for reading."""
        return self._opened

    def _frame_count(self) -> int:
        """Return the stream count, decoding a second handle if metadata lacks it."""
        if self._frame_count_cache is not None:
            return self._frame_count_cache

        count = int(getattr(self._stream, "frames", 0) or 0)
        if count <= 0 and not isinstance(self._source, int):
            container = av.open(str(self._source), mode="r")
            try:
                count = sum(1 for _ in container.decode(video=self._stream.index))
            finally:
                container.close()
        self._frame_count_cache = count
        return count

    def get(self, property_id: int) -> float:
        """Return the supported OpenCV capture property as a float."""
        if not self._opened:
            return 0.0
        if property_id == _CAP_PROP_FRAME_WIDTH:
            return float(self._stream.width)
        if property_id == _CAP_PROP_FRAME_HEIGHT:
            return float(self._stream.height)
        if property_id == _CAP_PROP_FPS:
            rate = getattr(self._stream, "average_rate", None) or getattr(
                self._stream, "base_rate", None
            )
            return float(rate) if rate is not None else 0.0
        if property_id == _CAP_PROP_FRAME_COUNT:
            return float(self._frame_count())
        if property_id == _CAP_PROP_POS_FRAMES:
            return float(self._position)
        return 0.0

    def _reset(self) -> None:
        """Seek the decoder to the first frame and reset the logical position."""
        self._container.seek(0, stream=self._stream, backward=True)
        self._frames = iter(self._container.decode(video=self._stream.index))
        self._position = 0

    def set(self, property_id: int, value: float) -> bool:
        """Set the supported frame-position property using exact frame decoding."""
        if not self._opened or property_id != _CAP_PROP_POS_FRAMES:
            return False
        target = max(0, round(value))
        if target > self._frame_count():
            return False
        if target < self._position:
            self._reset()
        while self._position < target:
            success, _ = self.read()
            if not success:
                return False
        return True

    def read(self) -> tuple[bool, npt.NDArray[np.uint8] | None]:
        """Decode and return the next frame in OpenCV's BGR array format."""
        if not self._opened or self._frames is None:
            return False, None
        try:
            frame = next(self._frames)
        except StopIteration:
            return False, None
        except Exception as exc:
            self._error = exc
            return False, None

        self._position += 1
        return True, frame.to_ndarray(format="bgr24")

    def grab(self) -> bool:
        """Decode and discard one frame."""
        success, _ = self.read()
        return success

    def release(self) -> None:
        """Close the PyAV container and make subsequent reads return false."""
        container = self._container
        self._container = None
        self._stream = None
        self._frames = None
        self._opened = False
        if container is not None:
            container.close()


class _VideoWriter:
    """Expose OpenCV-shaped video writing backed by PyAV encoding."""

    def __init__(
        self,
        filename: str | os.PathLike[str],
        fourcc: int,
        fps: float,
        frame_size: tuple[int, int],
        is_color: bool = True,
    ) -> None:
        """Open a PyAV writer for the requested codec and frame dimensions."""
        del is_color
        self._container: Any = None
        self._stream: Any = None
        self._width, self._height = frame_size
        self._opened = False
        self._error: Exception | None = None

        try:
            codec, pixel_format = _codec_details(fourcc)
            self._container = av.open(str(filename), mode="w")
            rate = Fraction(str(fps)).limit_denominator(100_000)
            self._stream = self._container.add_stream(codec, rate=rate)
            self._stream.width = self._width
            self._stream.height = self._height
            self._stream.pix_fmt = pixel_format
            self._opened = True
        except Exception as exc:
            self._error = exc
            self.release()

    def isOpened(self) -> bool:
        """Return whether the writer initialized successfully."""
        return self._opened

    def write(self, frame: npt.NDArray[np.uint8]) -> None:
        """Encode one BGR frame and mux all packets produced by the encoder."""
        if not self._opened or self._container is None or self._stream is None:
            raise RuntimeError("Video writer is not open") from self._error
        if frame.shape != (self._height, self._width, 3):
            raise ValueError(
                "Video frame must have shape "
                f"({self._height}, {self._width}, 3), got {frame.shape}"
            )
        if frame.dtype != np.uint8:
            raise ValueError("Video frames must use uint8 dtype")

        video_frame = av.VideoFrame.from_ndarray(
            np.ascontiguousarray(frame), format="bgr24"
        )
        for packet in self._stream.encode(video_frame):
            self._container.mux(packet)

    def release(self) -> None:
        """Flush delayed encoder packets and close the output container."""
        container = self._container
        stream = self._stream
        self._container = None
        self._stream = None
        self._opened = False
        if container is None:
            return
        try:
            if stream is not None:
                for packet in stream.encode():
                    container.mux(packet)
        finally:
            container.close()


def _copy_stream(container: Any, source_stream: Any) -> Any:
    """Create an output stream with the codec parameters needed for remuxing."""
    codec = source_stream.codec_context.name
    if source_stream.type == "video":
        rate = source_stream.average_rate or source_stream.base_rate
    else:
        rate = source_stream.sample_rate
    output_stream = container.add_stream(codec, rate=rate)

    if source_stream.type == "video":
        output_stream.width = source_stream.width
        output_stream.height = source_stream.height
        if source_stream.codec_context.format is not None:
            output_stream.pix_fmt = source_stream.codec_context.format.name
    else:
        output_stream.layout = source_stream.layout.name

    extradata = source_stream.codec_context.extradata
    if extradata:
        output_stream.codec_context.extradata = extradata
    return output_stream


def _timestamp_seconds(timestamp: int | None, time_base: Any) -> float | None:
    """Convert a stream timestamp to seconds while preserving missing values."""
    return None if timestamp is None else float(timestamp * time_base)


def _mux_audio(source_path: str, video_path: str) -> None:
    """Remux the source's first audio stream into the processed video with PyAV."""
    source_container: Any = None
    video_container: Any = None
    output_container: Any = None
    temporary_path: str | None = None
    try:
        source_container = av.open(source_path, mode="r")
        video_container = av.open(video_path, mode="r")
        if not source_container.streams.audio or not video_container.streams.video:
            logger.info("No audio or video stream available; leaving output unchanged")
            return

        source_audio = source_container.streams.audio[0]
        target_video = video_container.streams.video[0]
        suffix = Path(video_path).suffix
        with tempfile.NamedTemporaryFile(
            suffix=suffix, dir=str(Path(video_path).absolute().parent), delete=False
        ) as temporary_file:
            temporary_path = temporary_file.name

        output_container = av.open(temporary_path, mode="w")
        output_video = _copy_stream(output_container, target_video)
        output_audio = _copy_stream(output_container, source_audio)

        # Copy encoded packets to avoid a lossy decode/re-encode cycle for both streams.
        video_base: int | None = None
        video_duration: float | None = None
        for packet in video_container.demux(target_video):
            if packet.pts is None and packet.dts is None:
                continue
            if video_base is None:
                video_base = packet.pts if packet.pts is not None else packet.dts
            if video_base is None:
                continue
            packet.pts = None if packet.pts is None else packet.pts - video_base
            packet.dts = None if packet.dts is None else packet.dts - video_base
            packet.stream = output_video
            output_container.mux(packet)
            packet_end = packet.pts
            if packet_end is not None:
                packet_end += packet.duration or 0
                packet_seconds = _timestamp_seconds(packet_end, target_video.time_base)
                if packet_seconds is not None:
                    video_duration = max(video_duration or 0.0, packet_seconds)

        # Rebase audio independently and stop at the processed video duration, matching
        # ffmpeg's `-shortest` behavior without requiring a system executable.
        audio_base: int | None = None
        for packet in source_container.demux(source_audio):
            if packet.pts is None and packet.dts is None:
                continue
            if audio_base is None:
                audio_base = packet.pts if packet.pts is not None else packet.dts
            if audio_base is None:
                continue
            relative_pts = None if packet.pts is None else packet.pts - audio_base
            relative_seconds = _timestamp_seconds(relative_pts, source_audio.time_base)
            if (
                video_duration is not None
                and relative_seconds is not None
                and relative_seconds > video_duration
            ):
                break
            packet.pts = relative_pts
            packet.dts = None if packet.dts is None else packet.dts - audio_base
            packet.stream = output_audio
            output_container.mux(packet)

        output_container.close()
        output_container = None
        video_container.close()
        video_container = None
        source_container.close()
        source_container = None
        os.replace(temporary_path, video_path)
        temporary_path = None
    except Exception as exc:
        logger.warning("Audio remuxing failed: %s. Output video has no audio.", exc)
    finally:
        if output_container is not None:
            output_container.close()
        if video_container is not None:
            video_container.close()
        if source_container is not None:
            source_container.close()
        if temporary_path is not None and os.path.exists(temporary_path):
            os.remove(temporary_path)
