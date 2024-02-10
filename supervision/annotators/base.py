from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np
from PIL import Image

from supervision.detection.core import Detections

"""
An image of type `np.ndarray` or `PIL.Image.Image`.

Unlike a `Union`, ensures the type remains consistent. If a function
takes an `ImgType` argument and returns an `ImgType`, when you pass
an `np.ndarray`, you will always get an `np.ndarray` back.
"""
ImgType = TypeVar("ImgType", np.ndarray, Image.Image)


class BaseAnnotator(ABC):
    @abstractmethod
    def annotate(self, scene: ImgType, detections: Detections) -> ImgType:
        pass
