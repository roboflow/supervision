from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

from supervision.video.utils import VideoInfo


class BaseBackend(ABC):
    def __init__(self):
        self.cap = None
        self.video_info = None
        self.writer = None
        self.path = None

    @abstractmethod
    def get_sink(
        self, target_path: str, video_info: VideoInfo, codec: str = "mp4v"
    ) -> "BaseWriter":
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
    def release(self) -> None:
        pass

    @abstractmethod
    def save(
        self,
        target_path: str,
        callback: Callable[[np.ndarray, int], np.ndarray],
        fps: int | None = None,
        progress_message: str = "Processing video",
        show_progress: bool = False,
        codec: str = "mp4v",
    ):
        pass


class BaseWriter(ABC):
    @abstractmethod
    def write(self, frame: np.ndarray) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
