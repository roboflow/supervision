from abc import ABC, abstractmethod
from typing import Tuple

import av
import cv2
import numpy as np

from supervision.video.dataclasses import VideoInfo


class VideoBackend(ABC):
    @abstractmethod
    def __init__(self, source_path: str): ...

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray]: ...

    @abstractmethod
    def get_info(self) -> VideoInfo: ...

    @abstractmethod
    def seek(self, frame_idx: int): ...

    @abstractmethod
    def release(self) -> None: ...


class VideoWriter(ABC):
    @abstractmethod
    def __init__(self, target_path: str, video_info: VideoInfo): ...

    @abstractmethod
    def write(self, frame: np.ndarray): ...

    @abstractmethod
    def release(self) -> None: ...


class OpenCVBackend(VideoBackend):
    def __init__(self, source_path: str):
        self.source_path = source_path
        self.video = cv2.VideoCapture(source_path)
        if not self.video.isOpened():
            raise Exception(f"Could not open video at {source_path}")

    def read(self) -> tuple[bool, np.ndarray]:
        return self.video.read()

    def get_info(self) -> VideoInfo:
        width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.video.get(cv2.CAP_PROP_FPS))
        total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        return VideoInfo(width, height, fps, total_frames)

    def seek(self, frame_idx: int):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def release(self) -> None:
        self.video.release()


class OpenCVWriter(VideoWriter):
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

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.release()

    def write(self, frame: np.ndarray):
        self.__writer.write(frame)

    def release(self) -> None:
        self.__writer.release()


class PyAVBackend(VideoBackend):
    def __init__(self, source_path: str):
        self.source_path = source_path
        try:
            self.container = av.open(source_path)
        except av.AVError:
            raise Exception(f"Could not open video at {source_path}")

    def read(self) -> tuple[bool, np.ndarray]:
        try:
            frame = next(self.container.decode(video=0))
            return True, frame.to_ndarray(format="bgr24")
        except StopIteration:
            return False, None

    def get_info(self) -> VideoInfo:
        video_stream = self.container.streams.video[0]
        width = video_stream.width
        height = video_stream.height
        fps = video_stream.average_rate
        total_frames = video_stream.frames
        return VideoInfo(width, height, int(fps), total_frames)

    def seek(self, frame_idx: int):
        self.container.seek(frame_idx)

    def release(self) -> None:
        self.container.close()


class PyAVWriter(VideoWriter):
    def __init__(self, target_path: str, video_info: VideoInfo, codec: str = "libx264"):
        self.target_path = target_path
        self.video_info = video_info
        self.container = av.open(target_path, mode="w")
        self.stream = self.container.add_stream(codec, rate=video_info.fps)
        self.stream.width = video_info.width
        self.stream.height = video_info.height

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.release()

    def write(self, frame: np.ndarray):
        av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        for packet in self.stream.encode(av_frame):
            self.container.mux(packet)

    def release(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
