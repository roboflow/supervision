from __future__ import annotations

import os
import subprocess
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Generator
from dataclasses import dataclass
from enum import Enum

import cv2
import imageio_ffmpeg
import numpy as np
from tqdm.auto import tqdm

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()


class SOURCE_TYPE(Enum):
    VIDEO_FILE = "VIDEO_FILE"
    WEBCAM = "WEBCAM"
    RTSP = "RTSP"


@dataclass
class VideoInfo:
    """
    A class to store video information, including width, height, fps and
        total number of frames.

    Attributes:
        width (int): width of the video in pixels
        height (int): height of the video in pixels
        fps (int): frames per second of the video
        total_frames (Optional[int]): total number of frames in the video,
            default is None

    Examples:
        ```python
        import supervision as sv

        video_info = sv.VideoInfo.from_video_path(video_path=<SOURCE_VIDEO_FILE>)

        video_info
        # VideoInfo(width=3840, height=2160, fps=25, total_frames=538)

        video_info.resolution_wh
        # (3840, 2160)
        ```
    """

    width: int
    height: int
    fps: int
    total_frames: int | None = None
    source_type: SOURCE_TYPE | None = None

    @classmethod
    def from_video_path(cls, video_path: str) -> VideoInfo:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise Exception(f"Could not open video at {video_path}")

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(round(video.get(cv2.CAP_PROP_FPS)))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()
        return VideoInfo(width, height, fps, total_frames)

    @property
    def resolution_wh(self) -> tuple[int, int]:
        return self.width, self.height


class Backend(ABC):
    def __init__(self):
        self.cap = None
        self.video_info = None
        self.writer = None
        self.path = None

    @abstractmethod
    def get_sink(
        self, target_path: str, video_info: VideoInfo, codec: str = "mp4v"
    ) -> Writer:
        pass

    @abstractmethod
    def open(self, path: str) -> None:
        pass

    @abstractmethod
    def isOpened(self) -> bool:
        pass

    @abstractmethod
    def _set_video_info(self) -> VideoInfo:
        pass

    @abstractmethod
    def info(self) -> VideoInfo:
        pass

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray]:
        pass

    @abstractmethod
    def grab(self) -> bool:
        pass

    @abstractmethod
    def seek(self, frame_idx: int) -> None:
        pass

    @abstractmethod
    def release(self) -> None:
        pass

    @abstractmethod
    def frames(
        self,
        *,
        start: int = 0,
        end: int | None = None,
        stride: int = 1,
        resolution_wh: tuple[int, int] | None = None,
    ):
        pass

    @abstractmethod
    def save(
        self,
        target_path: str,
        callback: Callable[[np.ndarray, int], np.ndarray],
        fps: int | None = None,
        progress_message: str = "Processing video",
        show_progress: bool = False,
    ):
        pass


class Writer(ABC):
    @abstractmethod
    def write(self, frame: np.ndarray) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class OpenCVBackend(Backend):
    """
    OpenCV implementation of the Backend interface.
    Handles video capture, frame reading, seeking, and writing operations using OpenCV.
    """

    def __init__(self):
        """Initialize the OpenCV backend with empty video capture and writer objects."""
        super().__init__()
        self.cap = None
        self.video_info = None
        self.writer = None
        self.path = None

    def get_sink(self, target_path: str, video_info: VideoInfo, codec: str = "mp4v"):
        """Create a video writer for saving frames using OpenCV.

        Args:
            target_path (str): Path where the video will be saved.
            video_info (VideoInfo): Video information containing resolution and FPS.
            codec (str, optional): FourCC code for video codec. Defaults to "mp4v".

        Returns:
            OpenCVWriter: A video writer object for writing frames.
        """
        return OpenCVWriter(
            target_path, video_info.fps, video_info.resolution_wh, codec
        )

    def open(self, path: str) -> None:
        """
        Open a video source and initialize the video capture object.

        Args:
            path (str): Path to the video file, RTSP URL, or camera index.

        Raises:
            RuntimeError: If unable to open the video source.
            ValueError: If the source type is not supported.
        """
        self.cap = cv2.VideoCapture(path)
        self.path = path

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {path}")
        self.video_info = self._set_video_info()

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

    def isOpened(self) -> bool:
        """Check if the video source is opened successfully.

        Returns:
            bool: True if the video source is opened, False otherwise.
        """
        return self.cap.isOpened()

    def _set_video_info(self) -> VideoInfo:
        """Set up video information from the opened video source.

        Returns:
            VideoInfo: Object containing video properties like width, height, fps, etc.

        Raises:
            RuntimeError: If the video source is not opened yet.
        """
        if not self.isOpened():
            raise RuntimeError("Video not opened yet.")
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return VideoInfo(width, height, fps, total_frames)

    def info(self) -> VideoInfo:
        """Get video information.

        Returns:
            VideoInfo: Object containing video properties.

        Raises:
            RuntimeError: If the video source is not opened yet.
        """
        if not self.isOpened():
            raise RuntimeError("Video not opened yet.")
        return self.video_info

    def read(self) -> tuple[bool, np.ndarray]:
        """Read a frame from the video source.

        Returns:
            tuple[bool, np.ndarray]: A tuple containing:
                - bool: True if frame was successfully read
                - np.ndarray: The video frame in BGR format

        Raises:
            RuntimeError: If the video source is not opened yet.
        """
        if self.cap is None:
            raise RuntimeError("Video not opened yet.")
        ret, frame = self.cap.read()
        return ret, frame

    def grab(self) -> bool:
        """Grab a frame from video source without decoding.

        Returns:
            bool: True if frame was successfully grabbed.

        Raises:
            RuntimeError: If the video source is not opened yet.
        """
        if self.cap is None:
            raise RuntimeError("Video not opened yet.")
        return self.cap.grab()

    def seek(self, frame_idx: int) -> None:
        """Seek to a specific frame in the video.

        Args:
            frame_idx (int): Index of the frame to seek to (0-based).

        Raises:
            RuntimeError: If the video source is not opened yet.
        """
        if self.cap is None:
            raise RuntimeError("Video not opened yet.")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def release(self) -> None:
        """Release the video capture resources."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def frames(
        self,
        *,
        start: int = 0,
        end: int | None = None,
        stride: int = 1,
        resolution_wh: tuple[int, int] | None = None,
    ):
        """Generate frames from the video source.

        Args:
            start (int, optional): Starting frame index. Defaults to 0.
            end (int | None, optional): Ending frame index. Defaults to None.
        stride (int, optional): Number of frames to skip. Defaults to 1.
            resolution_wh (tuple[int, int] | None, optional): Target resolution
                (width, height). If provided, frames will be resized. Defaults to None.

            Yields:
                np.ndarray: Video frames in BGR format.        Raises:
            RuntimeError: If the video source is not opened yet.
        """
        if self.cap is None:
            raise RuntimeError("Video not opened yet.")

        total_frames = self.video_info.total_frames if self.video_info else 0
        is_live_stream = total_frames <= 0

        if is_live_stream:
            while True:
                for _ in range(stride - 1):
                    if not self.grab():
                        return
                ret, frame = self.read()
                if not ret:
                    return
                if resolution_wh is not None:
                    frame = cv2.resize(frame, resolution_wh)
                yield frame
        else:
            if end is None or end > total_frames:
                end = total_frames

            frame_idx = start
            while frame_idx < end:
                self.seek(frame_idx)
                ret, frame = self.read()
                if not ret:
                    break
                if resolution_wh is not None:
                    frame = cv2.resize(frame, resolution_wh)
                yield frame
                frame_idx += stride

    def save(
        self,
        target_path: str,
        callback: Callable[[np.ndarray, int], np.ndarray],
        fps: int | None = None,
        progress_message: str = "Processing video",
        show_progress: bool = False,
    ):
        """Save processed video frames to a file with audio preservation.

        Args:
            target_path (str): Path where the processed video will be saved.
            callback (Callable[[np.ndarray, int], np.ndarray]): Function that processes
                each frame. Takes frame and index as input, returns processed frame.
            fps (int | None, optional): Output video FPS. If None, uses source FPS.
            progress_message (str, optional): Message to show in progress bar.
            show_progress (bool, optional): Whether to show progress bar.

        Raises:
            RuntimeError: If video source is not opened.
            ValueError: If source is not a video file.
        """
        if self.cap is None:
            raise RuntimeError("Video not opened yet.")

        if self.video_info.source_type != SOURCE_TYPE.VIDEO_FILE:
            raise ValueError("Only video files can be saved.")

        if self.writer is not None:
            self.writer.close()
            self.writer = None

        source_codec = self.cap.get(cv2.CAP_PROP_FOURCC)

        if fps is None:
            fps = self.video_info.fps

        self.writer = OpenCVWriter(
            target_path, fps, self.video_info.resolution_wh, source_codec
        )
        total_frames = self.video_info.total_frames
        frames_generator = self.frames()
        for index, frame in enumerate(
            tqdm(
                frames_generator,
                total=total_frames,
                disable=not show_progress,
                desc=progress_message,
            )
        ):
            result_frame = callback(frame, index)
            self.writer.write(frame=result_frame)

        self.writer.close()

        def has_audio_stream(video_path):
            result = subprocess.run(
                [ffmpeg_path, "-i", video_path],
                stdout=subprocess.DEVNULL,
                text=True,
            )

            return "Audio:" in result.stderr

        if has_audio_stream(self.path):
            video_input = target_path
            audio_source = self.path
            temp_output = "temp_output.mp4"
            subprocess.run(
                [
                    ffmpeg_path,
                    "-i",
                    video_input,
                    "-i",
                    audio_source,
                    "-map",
                    "0:v",
                    "-map",
                    "1:a",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-shortest",
                    temp_output,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )

            os.replace(temp_output, video_input)


class OpenCVWriter(Writer):
    """A class to handle video writing operations using OpenCV's VideoWriter.

    This class provides an interface to write frames to a video file using OpenCV,
    with support for different codecs and automatic fallback to mp4v if the specified
    codec fails.
    """

    def __init__(
        self,
        filename: str,
        fps: float,
        frame_size: tuple[int, int],
        codec: str = "mp4v",
    ):
        """Initialize the video writer.

        Args:
            filename (str): Path to the output video file.
            fps (float): Frames per second for the output video.
            frame_size (tuple[int, int]): Width and height of the output video frames.
            codec (str, optional): FourCC code for the video codec. Defaults to "mp4v".

        Raises:
            RuntimeError: If the video writer cannot be initialized.
        """
        try:
            fourcc_int = cv2.VideoWriter_fourcc(*codec)
            self.writer = cv2.VideoWriter(filename, fourcc_int, fps, frame_size)
        except Exception:
            fourcc_int = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(filename, fourcc_int, fps, frame_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for file: {filename}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write(self, frame: np.ndarray) -> None:
        """Write a frame to the video file.

        Args:
            frame (np.ndarray): The frame to write, in BGR format.
        """
        self.writer.write(frame)

    def close(self) -> None:
        """Release the video writer resources."""
        self.writer.release()


class Video:
    """High-level interface for video operations.

    This class provides a convenient interface for video operations including
    reading frames, saving processed videos, and video information access.
    """

    info: VideoInfo
    source: str | int
    backend: Backend

    def __init__(
        self, source: str | int, info: VideoInfo | None = None, backend: str = "opencv"
    ):
        if backend == "opencv":
            self.backend = OpenCVBackend()

        self.backend.open(source)
        self.info = self.backend.video_info
        self.source = source

    def __iter__(self):
        """Make the Video class iterable over frames.

        Returns:
            Generator: A generator yielding video frames.
        """
        return self.backend.frames()

    def sink(self, target_path: str, info: VideoInfo, codec: str = "mp4v") -> Writer:
        """Create a video writer for saving frames.

        Args:
            target_path (str): Path where the video will be saved.
            info (VideoInfo): Video information containing resolution and FPS.
            codec (str, optional): FourCC code for video codec. Defaults to "mp4v".

        Returns:
            Writer: A video writer object for writing frames.
        """
        return self.backend.get_sink(target_path, info, codec)

    def frames(
        self,
        stride: int = 1,
        start: int = 0,
        end: int | None = None,
        resolution_wh: tuple[int, int] | None = None,
    ):
        """Generate frames from the video.

        Args:
            stride (int, optional): Number of frames to skip. Defaults to 1.
            start (int, optional): Starting frame index. Defaults to 0.
            end (int | None, optional): Ending frame index. Defaults to None.
            resolution_wh (tuple[int, int] | None, optional): Target resolution
                (width, height). If provided, frames will be resized. Defaults to None.

        Returns:
            Generator: A generator yielding video frames.
        """
        return self.backend.frames(
            stride=stride, start=start, end=end, resolution_wh=resolution_wh
        )

    def save(
        self,
        target_path: str,
        callback: Callable[[np.ndarray, int], np.ndarray],
        fps: int | None = None,
        progress_message: str = "Processing video",
        show_progress: bool = False,
    ):
        """Save processed video frames to a file.

        Args:
            target_path (str): Path where the processed video will be saved.
            callback (Callable[[np.ndarray, int], np.ndarray]): Function that processes
                each frame. Takes frame and index as input, returns processed frame.
            fps (int | None, optional): Output video FPS.
            progress_message (str, optional): Message to show in progress bar.
                Defaults to "Processing video".
            show_progress (bool, optional): Whether to show progress bar.
                Defaults to False.
        """
        self.backend.save(
            target_path=target_path,
            callback=callback,
            fps=fps,
            progress_message=progress_message,
            show_progress=show_progress,
        )


@DeprecationWarning
class VideoSink:
    """
    Context manager that saves video frames to a file using OpenCV.

    Attributes:
        target_path (str): The path to the output file where the video will be saved.
        video_info (VideoInfo): Information about the video resolution, fps,
            and total frame count.
        codec (str): FOURCC code for video format

    Example:
        ```python
        import supervision as sv

        video_info = sv.VideoInfo.from_video_path(<SOURCE_VIDEO_PATH>)
        frames_generator = sv.get_video_frames_generator(<SOURCE_VIDEO_PATH>)

        with sv.VideoSink(target_path=<TARGET_VIDEO_PATH>, video_info=video_info) as sink:
            for frame in frames_generator:
                sink.write_frame(frame=frame)
        ```
    """  # noqa: E501 // docs

    def __init__(self, target_path: str, video_info: VideoInfo, codec: str = "mp4v"):
        self.target_path = target_path
        self.video_info = video_info
        self.__codec = codec
        self.__writer = None

    def __enter__(self):
        try:
            self.__fourcc = cv2.VideoWriter_fourcc(*self.__codec)
        except TypeError as e:
            print(str(e) + ". Defaulting to mp4v...")
            self.__fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.__writer = cv2.VideoWriter(
            self.target_path,
            self.__fourcc,
            self.video_info.fps,
            self.video_info.resolution_wh,
        )
        return self

    def write_frame(self, frame: np.ndarray):
        """
        Writes a single video frame to the target video file.

        Args:
            frame (np.ndarray): The video frame to be written to the file. The frame
                must be in BGR color format.
        """
        self.__writer.write(frame)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__writer.release()


@DeprecationWarning
def _validate_and_setup_video(
    source_path: str, start: int, end: int | None, iterative_seek: bool = False
):
    video = cv2.VideoCapture(source_path)
    if not video.isOpened():
        raise Exception(f"Could not open video at {source_path}")
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if end is not None and end > total_frames:
        raise Exception("Requested frames are outbound")
    start = max(start, 0)
    end = min(end, total_frames) if end is not None else total_frames

    if iterative_seek:
        while start > 0:
            success = video.grab()
            if not success:
                break
            start -= 1
    elif start > 0:
        video.set(cv2.CAP_PROP_POS_FRAMES, start)

    return video, start, end


@DeprecationWarning
def get_video_frames_generator(
    source_path: str,
    stride: int = 1,
    start: int = 0,
    end: int | None = None,
    iterative_seek: bool = False,
) -> Generator[np.ndarray]:
    """
    Get a generator that yields the frames of the video.

    Args:
        source_path (str): The path of the video file.
        stride (int): Indicates the interval at which frames are returned,
            skipping stride - 1 frames between each.
        start (int): Indicates the starting position from which
            video should generate frames
        end (Optional[int]): Indicates the ending position at which video
            should stop generating frames. If None, video will be read to the end.
        iterative_seek (bool): If True, the generator will seek to the
            `start` frame by grabbing each frame, which is much slower. This is a
            workaround for videos that don't open at all when you set the `start` value.

    Returns:
        (Generator[np.ndarray, None, None]): A generator that yields the
            frames of the video.

    Examples:
        ```python
        import supervision as sv

        for frame in sv.get_video_frames_generator(source_path=<SOURCE_VIDEO_PATH>):
            ...
        ```
    """
    video, start, end = _validate_and_setup_video(
        source_path, start, end, iterative_seek
    )
    frame_position = start
    while True:
        success, frame = video.read()
        if not success or frame_position >= end:
            break
        yield frame
        for _ in range(stride - 1):
            success = video.grab()
            if not success:
                break
        frame_position += stride
    video.release()


@DeprecationWarning
def process_video(
    source_path: str,
    target_path: str,
    callback: Callable[[np.ndarray, int], np.ndarray],
    max_frames: int | None = None,
    show_progress: bool = False,
    progress_message: str = "Processing video",
) -> None:
    """
    Process a video file by applying a callback function on each frame
        and saving the result to a target video file.

    Args:
        source_path (str): The path to the source video file.
        target_path (str): The path to the target video file.
        callback (Callable[[np.ndarray, int], np.ndarray]): A function that takes in
            a numpy ndarray representation of a video frame and an
            int index of the frame and returns a processed numpy ndarray
            representation of the frame.
        max_frames (Optional[int]): The maximum number of frames to process.
        show_progress (bool): Whether to show a progress bar.
        progress_message (str): The message to display in the progress bar.

    Examples:
        ```python
        import supervision as sv

        def callback(scene: np.ndarray, index: int) -> np.ndarray:
            ...

        process_video(
            source_path=<SOURCE_VIDEO_PATH>,
            target_path=<TARGET_VIDEO_PATH>,
            callback=callback
        )
        ```
    """
    source_video_info = VideoInfo.from_video_path(video_path=source_path)
    video_frames_generator = get_video_frames_generator(
        source_path=source_path, end=max_frames
    )
    with VideoSink(target_path=target_path, video_info=source_video_info) as sink:
        total_frames = (
            min(source_video_info.total_frames, max_frames)
            if max_frames is not None
            else source_video_info.total_frames
        )
        for index, frame in enumerate(
            tqdm(
                video_frames_generator,
                total=total_frames,
                disable=not show_progress,
                desc=progress_message,
            )
        ):
            result_frame = callback(frame, index)
            sink.write_frame(frame=result_frame)
        else:
            for index, frame in enumerate(video_frames_generator):
                result_frame = callback(frame, index)
                sink.write_frame(frame=result_frame)

    def has_audio_stream(video_path):
        result = subprocess.run(
            [ffmpeg_path, "-i", video_path],
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True,
        )

        return "Audio:" in result.stderr

    if has_audio_stream(source_path):
        video_input = target_path
        audio_source = source_path
        temp_output = "temp_output.mp4"
        subprocess.run(
            [
                ffmpeg_path,
                "-i",
                video_input,
                "-i",
                audio_source,
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                temp_output,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        os.replace(temp_output, video_input)


class FPSMonitor:
    """
    A class for monitoring frames per second (FPS) to benchmark latency.
    """

    def __init__(self, sample_size: int = 30):
        """
        Args:
            sample_size (int): The maximum number of observations for latency
                benchmarking.

        Examples:
            ```python
            import supervision as sv

            frames_generator = sv.get_video_frames_generator(source_path=<SOURCE_FILE_PATH>)
            fps_monitor = sv.FPSMonitor()

            for frame in frames_generator:
                # your processing code here
                fps_monitor.tick()
                fps = fps_monitor.fps
            ```
        """  # noqa: E501 // docs
        self.all_timestamps = deque(maxlen=sample_size)

    @property
    def fps(self) -> float:
        """
        Computes and returns the average FPS based on the stored time stamps.

        Returns:
            float: The average FPS. Returns 0.0 if no time stamps are stored.
        """
        if not self.all_timestamps:
            return 0.0
        taken_time = self.all_timestamps[-1] - self.all_timestamps[0]
        return (len(self.all_timestamps)) / taken_time if taken_time != 0 else 0.0

    def tick(self) -> None:
        """
        Adds a new time stamp to the deque for FPS calculation.
        """
        self.all_timestamps.append(time.monotonic())

    def reset(self) -> None:
        """
        Clears all the time stamps from the deque.
        """
        self.all_timestamps.clear()
