import av
import numpy as np

from fractions import Fraction
from supervision.video.backend import BaseBackend, BaseWriter
from supervision.video.utils import VideoInfo, SOURCE_TYPE


class pyAVBackend(BaseBackend):
    """
    PyAV implementation of the Backend interface.
    Handles video capture, frame reading, seeking, and writing operations using PyAV.
    """

    def __init__(self):
        super().__init__()
        self.container = None
        self.stream = None
        self.writer = pyAVWriter
        self.frame_generator = None
        self.video_info = None
        self.current_frame_idx = 0  # Track current frame number in decoding
    
    def open(self, path: str) -> None:
        """Open and initialize a video source.

        Opens a video file, RTSP stream, or webcam and initializes all necessary
        components for video processing.

        Args:
            path (str): Path to video file, RTSP URL, or camera index.

        Raises:
            RuntimeError: If unable to open the video source.
            ValueError: If the source type is not supported.
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
        return self.container is not None and self.stream is not None

    def _set_video_info(self) -> VideoInfo:
        if not self.isOpened():
            raise RuntimeError("Video not opened yet.")

        width = self.stream.width
        height = self.stream.height
        fps = float(self.stream.average_rate or self.stream.guessed_rate)
        if fps <= 0:
            fps = 30  # Default to 30fps if invalid

        total_frames = self.stream.frames
        if total_frames == 0:
            total_frames = None

        return VideoInfo(width, height, round(fps), total_frames)

    def info(self) -> VideoInfo:
        if not self.isOpened():
            raise RuntimeError("Video not opened yet.")
        return self.video_info

    def read(self) -> tuple[bool, np.ndarray]:
        """Read the next frame from the video stream.

        Returns:
            tuple[bool, np.ndarray]: A tuple containing:
                - bool: True if frame was successfully read
                - np.ndarray: The video frame in BGR format (H, W, 3)

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
        """Grab the next frame packet without decoding.

        A lightweight operation that skips frame decoding, useful for
        quick frame navigation. Returns success status of the grab operation.

        Returns:
            bool: True if a frame was successfully grabbed, False otherwise.

        Raises:
            RuntimeError: If the video source is not opened.
        """
        if not self.isOpened():
            raise RuntimeError("Video not opened yet.")

        try:
            for packet in self.container.demux(video=0):
                if packet.stream.type == 'video':
                    return True
            return False
        except (StopIteration, av.error.EOFError):
            return False

    def seek(self, frame_idx: int) -> None:
        """Seek to a specific frame in the video.

        Performs frame-accurate seeking by navigating to the nearest keyframe and
        decoding forward to the exact target frame. The next read() call will
        return the target frame.

        Args:
            frame_idx (int): Target frame index (0-based) to seek to.

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

        self.container.seek(timestamp, stream=self.stream, any_frame=False, backward=True)
        self.frame_generator = self.container.decode(video=0)

        self.current_frame_idx = 0
        while True:
            try:
                frame = next(self.frame_generator)
            except (StopIteration, av.error.EOFError):
                break

            if getattr(frame, "time", None) is not None:
                self.current_frame_idx = int(round(frame.time * framerate))
            elif getattr(frame, "pts", None) is not None:
                self.current_frame_idx = int(round((frame.pts * time_base) * framerate))
            else:
                self.current_frame_idx += 1

            if self.current_frame_idx >= frame_idx:
                def _prepend_frame(first_frame, gen):
                    yield first_frame
                    yield from gen
                self.frame_generator = _prepend_frame(frame, self.frame_generator)
                break

    def release(self) -> None:
        """Release all resources associated with the video stream.

        Closes the video container and resets all internal state variables
        to ensure proper cleanup of resources.
        """
        if self.container:
            self.container.close()
            self.container = None
            self.stream = None
            self.frame_generator = None

class pyAVWriter(BaseWriter):
    def __init__(
        self,
        filename: str,
        fps: int,
        frame_size: tuple[int, int],
        codec: str = "h264",
        backend: pyAVBackend | None = None,
    ):        
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
            self.stream.codec_context.time_base = Fraction(1, fps)

            # Frame index for PTS
            self.frame_idx = 0
            
            self.audio_stream_out = None
            self.audio_packets = []
            if backend.audio_stream and backend.audio_src_container:
                audio_codec_name = backend.audio_stream.codec_context.name
                audio_rate = backend.audio_stream.codec_context.rate  # Can be None for some codecs
                self.audio_stream_out = self.container.add_stream(audio_codec_name, rate=audio_rate)
                for packet in backend.audio_src_container.demux(backend.audio_stream):
                    if packet.dts is not None:
                        self.audio_packets.append(packet)

        except Exception as e:
            raise RuntimeError(f"Cannot open video writer for file: {filename}") from e

    def write(self, frame: np.ndarray) -> None:
        # Convert BGR (OpenCV) to RGB for PyAV
        frame_rgb = frame[..., ::-1]
        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")

        av_frame.pts = self.frame_idx
        av_frame.time_base = self.stream.codec_context.time_base
        self.frame_idx += 1

        # Encode frame and mux packets immediately
        packets = self.stream.encode(av_frame)
        for packet in packets:
            self.container.mux(packet)

    def close(self) -> None:
        # Flush encoder by calling encode() with no frame, mux all packets
        packets = self.stream.encode()
        for packet in packets:
            self.container.mux(packet)

        if self.audio_stream_out:
            for packet in self.audio_packets:
                packet.stream = self.audio_stream_out
                self.container.mux(packet)

        self.container.close()
