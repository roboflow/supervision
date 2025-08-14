from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
from typing import Any, cast

import numpy as np

try:
    import av
except ImportError as e:
    raise ImportError(
        "The pyav backend is not installed, "
        "please install it using `pip install supervision[ffmpeg]`"
    ) from e

from supervision.video.backends.base import Writer
from supervision.video.utils import VideoInfo


class PyAVWriter:
    def __init__(
        self,
        container: av.container.output.OutputContainer,
        stream: av.video.stream.VideoStream,
        info: VideoInfo,
    ):
        """PyAV based video writer.

        Args:
            container: An opened ``av.open(..., mode="w")`` output container.
            stream: The created output video stream.
            info: Output video information used to validate or resize frames.
        """
        self._container = container
        self._stream = stream
        self.info = info

    def write(
        self,
        frame: np.ndarray,
        frame_number: int,
        callback: Any = None,
    ) -> None:
        """Encode a frame, applying an optional callback and resize if needed.

        Args:
            frame: Input frame array, expected BGR with shape (H, W, 3).
            frame_number: Sequential frame number being written.
            callback: Optional function ``(frame, frame_number) -> frame`` that
                transforms the frame before writing.
        """
        if callback is not None:
            frame = callback(frame, frame_number)

        # Create a VideoFrame from the ndarray. We assume BGR input for parity with OpenCV.
        vframe = av.VideoFrame.from_ndarray(frame, format="bgr24")

        # Ensure expected output dimensions and pixel format for the encoder.
        target_fmt = self._stream.pix_fmt or "yuv420p"
        if (
            vframe.width != self.info.width
            or vframe.height != self.info.height
            or vframe.format.name != target_fmt
        ):
            vframe = vframe.reformat(
                width=self.info.width, height=self.info.height, format=target_fmt
            )

        # Encode and mux packets.
        packets = self._stream.encode(vframe)
        for packet in packets:
            self._container.mux(packet)

    def close(self) -> None:
        """Flush and close the underlying output container."""
        # Flush encoder
        packets = self._stream.encode(None)
        for packet in packets:
            self._container.mux(packet)
        self._container.close()

    def __enter__(self) -> Writer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class PyAVBackend:
    def __init__(self, source_path: str | int):
        """Create a new backend for a source path or stream URL.

        Args:
            source_path: File path or stream URL. Integer webcam indexes are not
                handled portably by PyAV, so pass a device URL if you need a camera.
        """
        self.source_path = source_path
        if isinstance(source_path, int):
            raise ValueError(
                "Numeric webcam indexes are not supported by this PyAV backend. "
                "Provide a device URL, or use the OpenCV backend for webcams."
            )

        try:
            self.container: av.container.input.InputContainer = av.open(
                self.source_path, mode="r"
            )
        except Exception as e:
            raise ValueError(
                f"Could not open video source {self.source_path!r}: {e}"
            ) from e

        # Pick the first video stream.
        video_streams = [s for s in self.container.streams if s.type == "video"]
        if not video_streams:
            self.container.close()
            raise ValueError(f"No video stream found in source {self.source_path!r}")

        self.video_stream: av.video.stream.VideoStream = cast(
            av.video.stream.VideoStream, video_streams[0]
        )
        self.video_stream.thread_type = "AUTO"  # Improve performance on some inputs
        self._decoder = self.container.decode(video=self.video_stream.index)

    def _compute_fps(self) -> tuple[int, float]:
        avg = self.video_stream.average_rate
        if avg is None:
            avg = self.video_stream.guessed_rate
        if avg is None:
            precise = 30.0  # As a last resort, fall back to 30 fps
        else:
            # average_rate is a Fraction
            precise = float(avg)
        fps_int = round(precise) if precise > 0 else 30
        return fps_int, precise

    def info(self) -> VideoInfo:
        """Return static information (width, height, fps, precise_fps, total_frames)."""
        w = int(self.video_stream.codec_context.width or self.video_stream.width)
        h = int(self.video_stream.codec_context.height or self.video_stream.height)
        fps_int, precise = self._compute_fps()
        n = (
            int(self.video_stream.frames)
            if self.video_stream.frames not in (None, 0)
            else None
        )
        return VideoInfo(w, h, fps_int, precise, n)

    def _reset_decoder(self) -> None:
        self._decoder = self.container.decode(video=self.video_stream.index)

    def _next_avframe(self) -> av.VideoFrame | None:
        try:
            return next(self._decoder)
        except StopIteration:
            return None

    def read(self) -> tuple[bool, np.ndarray]:
        """Decode the next frame as a BGR numpy array."""
        frame = self._next_avframe()
        if frame is None:
            return False, np.empty((0, 0, 3), dtype=np.uint8)
        arr = frame.to_ndarray(format="bgr24")
        return True, arr

    def grab(self) -> bool:
        """Advance to the next frame without materializing pixel data."""
        return self._next_avframe() is not None

    def _pts_from_frame_index(self, frame_idx: int) -> int:
        fps_int, precise = self._compute_fps()
        time_base: Fraction = self.video_stream.time_base or Fraction(
            1, max(fps_int, 1)
        )
        # seconds to PTS units
        seconds = frame_idx / (precise if precise > 0 else max(fps_int, 1))
        pts = round(seconds / float(time_base))
        return pts

    def seek(self, frame_idx: int) -> None:
        """Seek so that the next call to ``read`` returns ``frame_idx``.

        This computes a timestamp from the frame index using the stream frame rate,
        performs an accurate seek on the video stream, then resets the decoder.
        """
        pts = self._pts_from_frame_index(frame_idx)
        # Use any_frame=False to seek to keyframes, then decoder will advance
        self.container.seek(
            pts, stream=self.video_stream, any_frame=False, backward=True
        )
        self._reset_decoder()

    # ----------- Writer factory -----------
    def writer(self, path: str, info: VideoInfo, codec: str | None = None) -> Writer:
        """Return a writer that encodes frames to a file path.

        Args:
            path: Target file path.
            info: Expected output resolution and fps.
            codec: FFmpeg encoder name. Examples: "libx264", "h264", "hevc", "mpeg4".
        """
        out = av.open(path, mode="w")
        enc = cast(
            av.video.stream.VideoStream,
            out.add_stream(codec or "libx264", rate=info.fps),
        )
        enc.width = info.width
        enc.height = info.height
        enc.pix_fmt = "yuv420p"  # Use a broadly compatible pixel format
        enc.options = {
            "movflags": "+faststart"
        }  # Improve default compatibility for MP4
        return cast(Writer, PyAVWriter(out, enc, info))

    def frames(
        self,
        stride: int = 1,
        start: int = 0,
        end: int | None = None,
        resolution_wh: tuple[int, int] | None = None,
        _interpolation: Any = None,  # Kept for API parity. PyAV scales internally.
    ) -> Iterator[np.ndarray]:
        """Yield frames lazily with optional skipping and resizing.

        Args:
            stride: Number of frames to skip between yielded frames. One yields every frame.
            start: First frame index to yield.
            end: Index after the last frame to yield. None means until exhaustion.
            resolution_wh: Optional (width, height) to resize each yielded frame to.
            interpolation: Ignored. Present only for API parity with the OpenCV backend.
        Yields:
            np.ndarray: The next decoded and optionally resized frame in BGR order.
        """
        if stride < 1:
            raise ValueError("stride must be >= 1")

        info = self.info()
        total = (
            info.total_frames if info.total_frames and info.total_frames > 0 else None
        )
        if end is None and total is not None:
            end = total
        if end is not None and (start < 0 or start >= end):
            return

        # Position decoder at the start frame
        self.seek(start)
        current_idx = start
        infinite_stream = end is None

        while infinite_stream or current_idx < end:  # type: ignore[operator]
            vf = self._next_avframe()
            if vf is None:
                break

            if resolution_wh is not None and (
                vf.width != resolution_wh[0] or vf.height != resolution_wh[1]
            ):
                vf = vf.reformat(
                    width=resolution_wh[0], height=resolution_wh[1], format="bgr24"
                )
                arr = vf.to_ndarray(format="bgr24")
            else:
                arr = vf.to_ndarray(format="bgr24")

            yield arr
            current_idx += 1

            # Efficiently skip stride - 1 frames
            skip = stride - 1
            while skip > 0 and (
                infinite_stream or (end is not None and current_idx < end)
            ):
                if not self.grab():
                    return
                current_idx += 1
                skip -= 1

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield successive frames until exhaustion."""
        while True:
            ok, frame = self.read()
            if not ok:
                break
            yield frame

    # ----------- Resource management -----------
    def release(self) -> None:
        """Close the input container."""
        try:
            self.container.close()
        finally:
            self._decoder = iter(())

    def __enter__(self) -> PyAVBackend:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()

    # ----------- Random access helpers -----------
    def __len__(self) -> int:
        n = self.info().total_frames
        if n is None or n < 0:
            raise TypeError("length is unknown for this stream")
        return n

    def __getitem__(self, index: int) -> np.ndarray:
        """Return the frame at the given index without disturbing current decode state."""
        # Open a temporary container to avoid changing the main decoder position
        try:
            with av.open(self.source_path, mode="r") as tmp:
                vstreams = [s for s in tmp.streams if s.type == "video"]
                if not vstreams:
                    raise IndexError(f"No video stream in source {self.source_path!r}")
                vs = vstreams[0]

                # Compute PTS and seek
                avg = vs.average_rate or vs.guessed_rate or Fraction(30, 1)
                time_base: Fraction = vs.time_base or Fraction(1, int(avg))
                seconds = index / float(avg)
                pts = round(seconds / float(time_base))
                tmp.seek(pts, stream=vs, any_frame=False, backward=True)

                for frame in tmp.decode(video=vs.index):
                    return frame.to_ndarray(format="bgr24")
        except Exception as e:
            raise IndexError(f"Failed to read frame {index}: {e}") from e

        raise IndexError(f"Failed to read frame {index}")


# Provide a consistent alias for the core loader
Backend = PyAVBackend
