from __future__ import annotations

from fractions import Fraction

try:
    import av
except ImportError:
    av = None
import numpy as np

from supervision.video.backend import BaseBackend, BaseWriter
from supervision.video.utils import SOURCE_TYPE, VideoInfo


class pyAVBackend(BaseBackend):
    """
    PyAV-based implementation of the `BaseBackend` interface.

    This backend handles video capture, frame reading, seeking, and writing
    operations using the PyAV library. Supports local video files, webcams,
    and RTSP streams.
    """

    def __init__(self):
        super().__init__()

        if av is None:
            raise RuntimeError(
                "PyAV (`av` module) is not installed. Run `pip install av`."
            )

        self.container = None
        self.stream = None
        self.writer = pyAVWriter
        self.frame_generator = None
        self.video_info = None
        self.current_frame_idx = 0

    def open(self, path: str) -> None:
        """
        Open and initialize a video source.

        This method opens a video file, RTSP stream, or webcam, and sets up
        the necessary components for decoding and reading frames.

        Args:
            path (str | int): Path to the video file, RTSP URL, or webcam index.

        Raises:
            RuntimeError: If the video source cannot be opened.
            ValueError: If the source type is unsupported.
        """
        try:
            self.container = av.open(path)
            self.audio_src_container = self.container
            self.stream = self.container.streams.video[0]
            self.stream.thread_type = "AUTO"
            self.cap = self.container

            self.frame_generator = self.container.decode(video=0)
            self.video_info = self._set_video_info()
            self.current_frame_idx = 0

            # If audio exists
            if len(self.container.streams.audio) > 0:
                self.audio_stream = self.container.streams.audio[0]
            else:
                self.audio_stream = None

            if isinstance(path, int):
                self.video_info.source_type = SOURCE_TYPE.WEBCAM
            elif isinstance(path, str):
                self.video_info.source_type = (
                    SOURCE_TYPE.RTSP
                    if path.lower().startswith("rtsp://")
                    else SOURCE_TYPE.VIDEO_FILE
                )
            else:
                raise ValueError("Unsupported source type")

        except Exception as e:
            raise RuntimeError(f"Cannot open video source: {path}") from e

    def isOpened(self) -> bool:
        """Check if the video source has been successfully opened."""
        return self.container is not None and self.stream is not None

    def _set_video_info(self) -> VideoInfo:
        """
        Extract video information from the opened source.

        Returns:
            VideoInfo: Object containing width, height, fps, and frame count.

        Raises:
            RuntimeError: If the video source is not opened.
        """
        if not self.isOpened():
            raise RuntimeError("Video not opened yet.")

        width = self.stream.width
        height = self.stream.height
        fps = float(self.stream.average_rate or self.stream.guessed_rate)
        if fps <= 0:
            fps = 30

        total_frames = self.stream.frames
        if total_frames == 0:
            total_frames = None

        return VideoInfo(width, height, round(fps), total_frames)

    def info(self) -> VideoInfo:
        """
        Retrieve video information.

        Returns:
            VideoInfo: Video properties for the opened source.

        Raises:
            RuntimeError: If the video source is not opened.
        """
        if not self.isOpened():
            raise RuntimeError("Video not opened yet.")
        return self.video_info

    def read(self) -> tuple[bool, np.ndarray]:
        """
        Read and decode the next frame from the video source.

        Returns:
            tuple[bool, np.ndarray]:
                - `bool`: True if a frame was read successfully, False if end of stream.
                - `np.ndarray`: Frame data in BGR format (H, W, 3).

        Raises:
            RuntimeError: If the video source is not opened.
        """
        if not self.isOpened():
            raise RuntimeError("Video not opened yet.")

        try:
            frame = next(self.frame_generator)
            self.current_frame_idx += 1
            frame_bgr = frame.to_ndarray(format="bgr24")
            return True, frame_bgr
        except (StopIteration, av.error.EOFError):
            return False, np.array([])

    def grab(self) -> bool:
        """
        Grab the next frame packet without decoding it.

        Useful for skipping frames quickly without the overhead of decoding.

        Returns:
            bool: True if a frame packet was grabbed successfully, False otherwise.

        Raises:
            RuntimeError: If the video source is not opened.
        """
        if not self.isOpened():
            raise RuntimeError("Video not opened yet.")

        try:
            for packet in self.container.demux(video=0):
                if packet.stream.type == "video":
                    return True
            return False
        except (StopIteration, av.error.EOFError):
            return False

    def seek(self, frame_idx: int) -> None:
        """
        Seek to a specific frame index in the video.

        This uses keyframe-based seeking, then decodes forward to the exact
        requested frame.

        Args:
            frame_idx (int): Zero-based index of the target frame.

        Raises:
            RuntimeError: If the video source is not opened.
        """
        if not self.isOpened():
            raise RuntimeError("Video not opened yet.")

        framerate = float(self.stream.average_rate or self.stream.guessed_rate or 30.0)
        if framerate <= 0:
            framerate = 30.0

        time_base = float(self.stream.time_base)
        timestamp = int((frame_idx / framerate) / time_base)

        self.container.seek(
            timestamp, stream=self.stream, any_frame=False, backward=True
        )
        self.frame_generator = self.container.decode(video=0)

        self.current_frame_idx = 0
        while True:
            try:
                frame = next(self.frame_generator)
            except (StopIteration, av.error.EOFError):
                break

            if getattr(frame, "time", None) is not None:
                self.current_frame_idx = round(frame.time * framerate)
            elif getattr(frame, "pts", None) is not None:
                self.current_frame_idx = round((frame.pts * time_base) * framerate)
            else:
                self.current_frame_idx += 1

            if self.current_frame_idx >= frame_idx:

                def _prepend_frame(first_frame, gen):
                    yield first_frame
                    yield from gen

                self.frame_generator = _prepend_frame(frame, self.frame_generator)
                break

    def release(self) -> None:
        """
        Release the video source and free all associated resources.
        """
        if self.container:
            self.container.close()
            self.container = None
            self.stream = None
            self.frame_generator = None


class pyAVWriter(BaseWriter):
    """
    PyAV-based video writer.

    Writes frames to a video file with optional audio from a backend source.
    """

    def __init__(
        self,
        filename: str,
        fps: int,
        frame_size: tuple[int, int],
        codec: str = "h264",
        backend: pyAVBackend | None = None,
    ):
        """
        Initialize a video writer.

        Args:
            filename (str): Output video file path.
            fps (int): Frames per second for the output video.
            frame_size (tuple[int, int]): Frame dimensions as (width, height).
            codec (str, optional): Video codec (default: "h264").
            backend (pyAVBackend, optional): Backend providing audio stream.

        Raises:
            RuntimeError: If the output file cannot be created.
        """
        try:
            self.container = av.open(filename, mode="w")
            self.backend = backend

            if codec is None:
                codec = "h264"
            self.stream = self.container.add_stream(codec, rate=fps)
            self.stream.width = frame_size[0]
            self.stream.height = frame_size[1]
            self.stream.pix_fmt = "yuv420p"

            # Set time_base explicitly for correct timing
            print(fps)
            self.stream.codec_context.time_base = Fraction(1, fps)

            # Frame index for PTS
            self.frame_idx = 0

            self.audio_stream_out = None
            self.audio_packets = []
            if backend and backend.audio_stream and backend.audio_src_container:
                audio_codec_name = backend.audio_stream.codec_context.name
                audio_rate = backend.audio_stream.codec_context.rate
                self.audio_stream_out = self.container.add_stream(
                    audio_codec_name, rate=audio_rate
                )
                for packet in backend.audio_src_container.demux(backend.audio_stream):
                    if packet.dts is not None:
                        self.audio_packets.append(packet)

        except Exception as e:
            raise RuntimeError(f"Cannot open video writer for file: {filename}") from e

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write(self, frame: np.ndarray) -> None:
        """
        Write a single frame to the output video.

        Args:
            frame (np.ndarray): Frame in BGR format (H, W, 3).
        """
        frame_rgb = frame[..., ::-1]
        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")

        av_frame.pts = self.frame_idx
        av_frame.time_base = self.stream.codec_context.time_base
        self.frame_idx += 1

        packets = self.stream.encode(av_frame)
        for packet in packets:
            self.container.mux(packet)

    def close(self) -> None:
        """
        Finalize the video file and close the writer.
        """
        packets = self.stream.encode()
        for packet in packets:
            self.container.mux(packet)

        if self.audio_stream_out:
            for packet in self.audio_packets:
                packet.stream = self.audio_stream_out
                self.container.mux(packet)

        self.container.close()
