from __future__ import annotations

from typing import Literal, Union

from supervision.video.backend.base import BaseBackend, BaseWriter
from supervision.video.backend.openCV import OpenCVBackend, OpenCVWriter
from supervision.video.backend.pyAV import pyAVBackend, pyAVWriter

BackendLiteral = Literal["opencv", "pyav"]
BackendTypes = Union[OpenCVBackend, pyAVBackend]
WriterTypes = Union[OpenCVWriter, pyAVWriter]

_backends = {
    "opencv": OpenCVBackend,
    "pyav": pyAVBackend,
}


def getBackend(backend: str) -> BaseBackend:
    if backend.lower() in _backends:
        return _backends[backend.lower()]()
    else:
        raise ValueError(f"Unsupported backend: {backend}")


__all__ = [
    "BackendLiteral",
    "BackendType",
    "BaseBackend",
    "BaseWriter",
    "OpenCVBackend",
    "OpenCVWriter",
    "getBackend",
    "pyAVBackend",
    "pyAVWriter",
]
