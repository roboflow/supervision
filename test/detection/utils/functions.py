import random

import numpy as np


def generate_boxes(
    n: int,
    W: int = 1920,
    H: int = 1080,
    min_size: int = 20,
    max_size: int = 200,
    seed: int | None = 1,
):
    """
    Generate N valid bounding boxes of format [x_min, y_min, x_max, y_max].

    Args:
        n (int): Number of boexs to generate
        W (int): Image width
        H (int): Image height
        min_size (int): Minimum box size (width/height)
        max_size (int): Maximum box size (width/height)
        seed (int | None): Random seed for reproducibility

    Returns:
        list[list[float]] | np.ndarray: List of boxes
    """
    random.seed(seed)
    boxes = []
    for _ in range(n):
        w = random.uniform(min_size, max_size)
        h = random.uniform(min_size, max_size)
        x1 = random.uniform(0, W - w)
        y1 = random.uniform(0, H - h)
        x2 = x1 + w
        y2 = y1 + h
        boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float32)
