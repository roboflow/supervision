from typing import TypeVar

import numpy as np
from PIL import Image

ImageType = TypeVar("ImageType", np.ndarray, Image.Image)
"""
An image of type `np.ndarray` or `PIL.Image.Image`.

Unlike a `Union`, ensures the type remains consistent. If a function
takes an `ImageType` argument and returns an `ImageType`, when you
pass an `np.ndarray`, you will get an `np.ndarray` back.
"""
