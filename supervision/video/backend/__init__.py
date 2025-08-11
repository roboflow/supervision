from __future__ import annotations

from typing import Literal, Union
from enum import Enum

from supervision.video.backend.opencv import OpenCVBackend, OpenCVWriter
from supervision.video.backend.pyav import pyAVBackend, pyAVWriter

BackendTypes = Union[OpenCVBackend, pyAVBackend]
WriterTypes = Union[OpenCVWriter, pyAVWriter]

class Backend(Enum):
    """
    Enumeration of Backends.
    """

    PYAV = "pyav"
    OPENCV = "opencv"

    @classmethod  
    def list(cls):  
        return list(map(lambda c: c.value, cls))  

    @classmethod  
    def from_value(cls, value: Backend | str) -> Backend:  
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

BackendDict = {
    Backend.PYAV: pyAVBackend,
    Backend.OPENCV: OpenCVBackend,
}

WriterDict = {
    Backend.PYAV: pyAVWriter,
    Backend.OPENCV: OpenCVWriter,
}

__all__ = [
    "BackendType",
    "WriterType",
    "Backend",
    "BackendDict",
    "WriterDict",
]
