from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from supervision.video.utils import VideoInfo


class BaseBackend(ABC):
    def __init__(self):
        self.cap = None
        self.video_info = None
        self.writer = None
        self.path = None

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

class BaseWriter(ABC):
    @abstractmethod
    def __init__(
        self,
        filename: str,
        fps: int,
        frame_size: tuple[int, int],
        codec: str | None = None,
        backend: BaseBackend | None = None,
    ):
        pass
    
    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @abstractmethod
    def write(self, frame: np.ndarray) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
