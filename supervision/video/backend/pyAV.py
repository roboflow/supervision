import av
import numpy as np

from fractions import Fraction
from supervision.video.backend.base import BaseBackend, BaseWriter
from supervision.video.utils import VideoInfo, SOURCE_TYPE


class pyAVWriter(BaseWriter):
    def __init__(
        self,
        filename: str,
        fps: int,
        frame_size: tuple[int, int],
        codec: str = "h264",
    ):
        try:
            self.container = av.open(filename, mode="w")

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

        self.container.close()

class pyAVBackend(BaseBackend):


    def __init__(self):
        super().__init__()
        self.container = None
        self.stream = None
        self.writer = pyAVWriter
        self.frame_generator = None
        self.video_info = None
        self.current_frame_idx = 0 

    def open(self, path: str) -> None:
    
        try:
            self.container = av.open(path)
            self.stream = self.container.streams.video[0]
            self.stream.thread_type = "AUTO"

            # cap is used for internals
            self.cap = self.container

            self.frame_generator = self.container.decode(video=0)
            self.video_info = self._set_video_info()
            self.current_frame_idx = 0

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
    
        if self.container:
            self.container.close()
            self.container = None
            self.stream = None
            self.frame_generator = None
