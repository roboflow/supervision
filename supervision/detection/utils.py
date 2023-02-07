from typing import Tuple

import cv2
import numpy as np


def generate_2d_mask(polygon: np.ndarray, resolution_wh: Tuple[int, int]) -> np.ndarray:
    """Generate a 2D mask from a polygon.

    Properties:
        polygon (np.ndarray): The polygon for which the mask should be generated, given as a list of vertices.
        resolution_wh (Tuple[int, int]): The width and height of the desired resolution.

    Returns:
        np.ndarray: The generated 2D mask, where the polygon is marked with `1`'s and the rest is filled with `0`'s.
    """
    width, height = resolution_wh
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], color=1)
    return mask
