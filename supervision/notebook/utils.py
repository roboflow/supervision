from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_frame_in_notebook(
    frame: np.ndarray, size: Tuple[int, int] = (10, 10), cmap: str = "gray"
):
    """
    Display a frame in Jupyter Notebook using Matplotlib

    Attributes:
        frame (np.ndarray): The frame to be displayed.
        size (Tuple[int, int]): The size of the plot. default:(10,10)
        cmap (str): the colormap to use for single channel images. default:gray

    Examples:
        ```python
        >>> from supervision.notebook.utils import show_frame_in_notebook

        %matplotlib inline
        show_frame_in_notebook(frame, (16, 16))
        ```
    """
    if frame.ndim == 2:
        plt.figure(figsize=size)
        plt.imshow(frame, cmap=cmap)
    else:
        plt.figure(figsize=size)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.show()
