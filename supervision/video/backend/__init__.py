from __future__ import annotations

from enum import Enum
from typing import Literal, Union

from supervision.video.backend.opencv import OpenCVBackend, OpenCVWriter
from supervision.video.backend.pyav import pyAVBackend, pyAVWriter

VideoBackendTypes = Union[OpenCVBackend, pyAVBackend]
VideoWriterTypes = Union[OpenCVWriter, pyAVWriter]


class VideoBackendType(Enum):
    """
    Enumeration of Backends.

    Attributes:
        PYAV (str): PyAV backend (powered by FFmpeg, supports audio rendering)
        OPENCV (str): OpenCV backend

    """

    PYAV = "pyav"
    OPENCV = "opencv"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def from_value(cls, value: VideoBackendType | str) -> VideoBackendType:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            value = value.lower()
            try:
                return cls(value)
            except ValueError:
                raise ValueError(f"Invalid value: {value}. Must be one of {cls.list()}")
        raise ValueError(
            f"Invalid value type: {type(value)}. Must be an instance of "
            f"{cls.__name__} or str."
        )


VideoBackendDict = {
    VideoBackendType.PYAV: pyAVBackend,
    VideoBackendType.OPENCV: OpenCVBackend,
}

VideoWriterDict = {
    VideoBackendType.PYAV: pyAVWriter,
    VideoBackendType.OPENCV: OpenCVWriter,
}

__all__ = [
    "VideoBackendDict",
    "VideoBackendType",
    "VideoBackendTypes",
    "VideoWriterDict",
    "VideoWriterTypes",
]
