from __future__ import annotations

from typing import Literal, overload, TypeVar, Union

from supervision.video.backend.base import BaseBackend, BaseWriter
from supervision.video.backend.openCV import OpenCVBackend, OpenCVWriter
from supervision.video.backend.pyAV import pyAVBackend, pyAVWriter

BackendT = TypeVar('BackendT', bound=BaseBackend)
BackendLiteral = Literal["opencv", "pyav"]
BackendType = Union[OpenCVBackend, pyAVBackend]

@overload
def getBackend(backend: Literal["opencv"]) -> OpenCVBackend:
    ...

@overload
def getBackend(backend: Literal["pyav"]) -> pyAVBackend:
    ...

def getBackend(backend: str) -> BaseBackend:
    if backend == "opencv":
        return OpenCVBackend()
    elif backend == "pyav":
        return pyAVBackend()
    else:
        raise ValueError(f"Unsupported backend: {backend}")

__all__ = [
    "BaseBackend",
    "BaseWriter",
    "OpenCVBackend",
    "OpenCVWriter",
    "pyAVBackend",
    "pyAVWriter",
    "getBackend",
    "BackendT",
    "BackendLiteral",
    "BackendType"
]