import numpy as np

import cv2

from supervision.annotators.dataclasses import Color
from supervision.commons.dataclasses import Rect


def draw_rect(image: np.ndarray, rect: Rect, color: Color, thickness: int) -> np.ndarray:
    cv2.rectangle()
