from __future__ import annotations

import itertools
import math
import os
import shutil
from collections.abc import Callable
from functools import partial
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt

from supervision.annotators.base import ImageType
from supervision.draw.color import Color, unify_to_bgr
from supervision.draw.utils import calculate_optimal_text_scale, draw_text
from supervision.geometry.core import Point
from supervision.utils.conversion import (
    cv2_to_pillow,
    ensure_cv2_image_for_processing,
    images_to_cv2,
)
from supervision.utils.iterables import create_batches, fill

RelativePosition = Literal["top", "bottom"]

MAX_COLUMNS_FOR_SINGLE_ROW_GRID = 3


@ensure_cv2_image_for_processing
def crop_image(
    image: ImageType,
    xyxy: npt.NDArray[int] | list[int] | tuple[int, int, int, int],
) -> ImageType:
    """
    Crops the given image based on the given bounding box.

    Args:
        image (ImageType): The image to be cropped. `ImageType` is a flexible type,
            accepting either `numpy.ndarray` or `PIL.Image.Image`.
        xyxy (Union[np.ndarray, List[int], Tuple[int, int, int, int]]): A bounding box
            coordinates in the format `(x_min, y_min, x_max, y_max)`, accepted as either
            a `numpy.ndarray`, a `list`, or a `tuple`.

    Returns:
        (ImageType): The cropped image. The type is determined by the input type and
            may be either a `numpy.ndarray` or `PIL.Image.Image`.

    === "OpenCV"

        ```python
        import cv2
        import supervision as sv

        image = cv2.imread(<SOURCE_IMAGE_PATH>)
        image.shape
        # (1080, 1920, 3)

        xyxy = [200, 400, 600, 800]
        cropped_image = sv.crop_image(image=image, xyxy=xyxy)
        cropped_image.shape
        # (400, 400, 3)
        ```

    === "Pillow"

        ```python
        from PIL import Image
        import supervision as sv

        image = Image.open(<SOURCE_IMAGE_PATH>)
        image.size
        # (1920, 1080)

        xyxy = [200, 400, 600, 800]
        cropped_image = sv.crop_image(image=image, xyxy=xyxy)
        cropped_image.size
        # (400, 400)
        ```

    ![crop_image](https://media.roboflow.com/supervision-docs/crop-image.png){ align=center width="800" }
    """  # noqa E501 // docs

    if isinstance(xyxy, (list, tuple)):
        xyxy = np.array(xyxy)
    xyxy = np.round(xyxy).astype(int)
    x_min, y_min, x_max, y_max = xyxy.flatten()
    return image[y_min:y_max, x_min:x_max]


@ensure_cv2_image_for_processing
def scale_image(image: ImageType, scale_factor: float) -> ImageType:
    """
    Scales the given image based on the given scale factor.

    Args:
        image (ImageType): The image to be scaled. `ImageType` is a flexible type,
            accepting either `numpy.ndarray` or `PIL.Image.Image`.
        scale_factor (float): The factor by which the image will be scaled. Scale
            factor > `1.0` zooms in, < `1.0` zooms out.

    Returns:
        (ImageType): The scaled image. The type is determined by the input type and
            may be either a `numpy.ndarray` or `PIL.Image.Image`.

    Raises:
        ValueError: If the scale factor is non-positive.

    === "OpenCV"

        ```python
        import cv2
        import supervision as sv

        image = cv2.imread(<SOURCE_IMAGE_PATH>)
        image.shape
        # (1080, 1920, 3)

        scaled_image = sv.scale_image(image=image, scale_factor=0.5)
        scaled_image.shape
        # (540, 960, 3)
        ```

    === "Pillow"

        ```python
        from PIL import Image
        import supervision as sv

        image = Image.open(<SOURCE_IMAGE_PATH>)
        image.size
        # (1920, 1080)

        scaled_image = sv.scale_image(image=image, scale_factor=0.5)
        scaled_image.size
        # (960, 540)
        ```
    """
    if scale_factor <= 0:
        raise ValueError("Scale factor must be positive.")

    width_old, height_old = image.shape[1], image.shape[0]
    width_new = int(width_old * scale_factor)
    height_new = int(height_old * scale_factor)
    return cv2.resize(image, (width_new, height_new), interpolation=cv2.INTER_LINEAR)


@ensure_cv2_image_for_processing
def resize_image(
    image: ImageType,
    resolution_wh: tuple[int, int],
    keep_aspect_ratio: bool = False,
) -> ImageType:
    """
    Resizes the given image to a specified resolution. Can maintain the original aspect
    ratio or resize directly to the desired dimensions.

    Args:
        image (ImageType): The image to be resized. `ImageType` is a flexible type,
            accepting either `numpy.ndarray` or `PIL.Image.Image`.
        resolution_wh (Tuple[int, int]): The target resolution as
            `(width, height)`.
        keep_aspect_ratio (bool): Flag to maintain the image's original
            aspect ratio. Defaults to `False`.

    Returns:
        (ImageType): The resized image. The type is determined by the input type and
            may be either a `numpy.ndarray` or `PIL.Image.Image`.

    === "OpenCV"

        ```python
        import cv2
        import supervision as sv

        image = cv2.imread(<SOURCE_IMAGE_PATH>)
        image.shape
        # (1080, 1920, 3)

        resized_image = sv.resize_image(
            image=image, resolution_wh=(1000, 1000), keep_aspect_ratio=True
        )
        resized_image.shape
        # (562, 1000, 3)
        ```

    === "Pillow"

        ```python
        from PIL import Image
        import supervision as sv

        image = Image.open(<SOURCE_IMAGE_PATH>)
        image.size
        # (1920, 1080)

        resized_image = sv.resize_image(
            image=image, resolution_wh=(1000, 1000), keep_aspect_ratio=True
        )
        resized_image.size
        # (1000, 562)
        ```

    ![resize_image](https://media.roboflow.com/supervision-docs/resize-image.png){ align=center width="800" }
    """  # noqa E501 // docs
    if keep_aspect_ratio:
        image_ratio = image.shape[1] / image.shape[0]
        target_ratio = resolution_wh[0] / resolution_wh[1]
        if image_ratio >= target_ratio:
            width_new = resolution_wh[0]
            height_new = int(resolution_wh[0] / image_ratio)
        else:
            height_new = resolution_wh[1]
            width_new = int(resolution_wh[1] * image_ratio)
    else:
        width_new, height_new = resolution_wh

    return cv2.resize(image, (width_new, height_new), interpolation=cv2.INTER_LINEAR)


@ensure_cv2_image_for_processing
def letterbox_image(
    image: ImageType,
    resolution_wh: tuple[int, int],
    color: tuple[int, int, int] | Color = Color.BLACK,
) -> ImageType:
    """
    Resizes and pads an image to a specified resolution with a given color, maintaining
    the original aspect ratio.

    Args:
        image (ImageType): The image to be resized. `ImageType` is a flexible type,
            accepting either `numpy.ndarray` or `PIL.Image.Image`.
        resolution_wh (Tuple[int, int]): The target resolution as
            `(width, height)`.
        color (Union[Tuple[int, int, int], Color]): The color to pad with. If tuple
            provided it should be in BGR format.

    Returns:
        (ImageType): The resized image. The type is determined by the input type and
            may be either a `numpy.ndarray` or `PIL.Image.Image`.

    === "OpenCV"

        ```python
        import cv2
        import supervision as sv

        image = cv2.imread(<SOURCE_IMAGE_PATH>)
        image.shape
        # (1080, 1920, 3)

        letterboxed_image = sv.letterbox_image(image=image, resolution_wh=(1000, 1000))
        letterboxed_image.shape
        # (1000, 1000, 3)
        ```

    === "Pillow"

        ```python
        from PIL import Image
        import supervision as sv

        image = Image.open(<SOURCE_IMAGE_PATH>)
        image.size
        # (1920, 1080)

        letterboxed_image = sv.letterbox_image(image=image, resolution_wh=(1000, 1000))
        letterboxed_image.size
        # (1000, 1000)
        ```

    ![letterbox_image](https://media.roboflow.com/supervision-docs/letterbox-image.png){ align=center width="800" }
    """  # noqa E501 // docs
    assert isinstance(image, np.ndarray)
    color = unify_to_bgr(color=color)
    resized_image = resize_image(
        image=image, resolution_wh=resolution_wh, keep_aspect_ratio=True
    )
    height_new, width_new = resized_image.shape[:2]
    padding_top = (resolution_wh[1] - height_new) // 2
    padding_bottom = resolution_wh[1] - height_new - padding_top
    padding_left = (resolution_wh[0] - width_new) // 2
    padding_right = resolution_wh[0] - width_new - padding_left
    image_with_borders = cv2.copyMakeBorder(
        resized_image,
        padding_top,
        padding_bottom,
        padding_left,
        padding_right,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    if image.shape[2] == 4:
        image[:padding_top, :, 3] = 0
        image[height_new - padding_bottom :, :, 3] = 0
        image[:, :padding_left, 3] = 0
        image[:, width_new - padding_right :, 3] = 0

    return image_with_borders


def overlay_image(
    image: npt.NDArray[np.uint8],
    overlay: npt.NDArray[np.uint8],
    anchor: tuple[int, int],
) -> npt.NDArray[np.uint8]:
    """
    Places an image onto a scene at a given anchor point, handling cases where
    the image's position is partially or completely outside the scene's bounds.

    Args:
        image (np.ndarray): The background scene onto which the image is placed.
        overlay (np.ndarray): The image to be placed onto the scene.
        anchor (Tuple[int, int]): The `(x, y)` coordinates in the scene where the
            top-left corner of the image will be placed.

    Returns:
        (np.ndarray): The result image with overlay.

    Examples:
        ```python
        import cv2
        import numpy as np
        import supervision as sv

        image = cv2.imread(<SOURCE_IMAGE_PATH>)
        overlay = np.zeros((400, 400, 3), dtype=np.uint8)
        result_image = sv.overlay_image(image=image, overlay=overlay, anchor=(200, 400))
        ```

    ![overlay_image](https://media.roboflow.com/supervision-docs/overlay-image.png){ align=center width="800" }
    """  # noqa E501 // docs
    scene_height, scene_width = image.shape[:2]
    image_height, image_width = overlay.shape[:2]
    anchor_x, anchor_y = anchor

    is_out_horizontally = anchor_x + image_width <= 0 or anchor_x >= scene_width
    is_out_vertically = anchor_y + image_height <= 0 or anchor_y >= scene_height

    if is_out_horizontally or is_out_vertically:
        return image

    x_min = max(anchor_x, 0)
    y_min = max(anchor_y, 0)
    x_max = min(scene_width, anchor_x + image_width)
    y_max = min(scene_height, anchor_y + image_height)

    crop_x_min = max(-anchor_x, 0)
    crop_y_min = max(-anchor_y, 0)
    crop_x_max = image_width - max((anchor_x + image_width) - scene_width, 0)
    crop_y_max = image_height - max((anchor_y + image_height) - scene_height, 0)

    if overlay.shape[2] == 4:
        b, g, r, alpha = cv2.split(
            overlay[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        )
        alpha = alpha[:, :, None] / 255.0
        overlay_color = cv2.merge((b, g, r))

        roi = image[y_min:y_max, x_min:x_max]
        roi[:] = roi * (1 - alpha) + overlay_color * alpha
        image[y_min:y_max, x_min:x_max] = roi
    else:
        image[y_min:y_max, x_min:x_max] = overlay[
            crop_y_min:crop_y_max, crop_x_min:crop_x_max
        ]

    return image


class ImageSink:
    def __init__(
        self,
        target_dir_path: str,
        overwrite: bool = False,
        image_name_pattern: str = "image_{:05d}.png",
    ):
        """
        Initialize a context manager for saving images.

        Args:
            target_dir_path (str): The target directory where images will be saved.
            overwrite (bool): Whether to overwrite the existing directory.
                Defaults to False.
            image_name_pattern (str): The image file name pattern.
                Defaults to "image_{:05d}.png".

        Examples:
            ```python
            import supervision as sv

            frames_generator = sv.get_video_frames_generator(<SOURCE_VIDEO_PATH>, stride=2)

            with sv.ImageSink(target_dir_path=<TARGET_CROPS_DIRECTORY>) as sink:
                for image in frames_generator:
                    sink.save_image(image=image)
            ```
        """  # noqa E501 // docs

        self.target_dir_path = target_dir_path
        self.overwrite = overwrite
        self.image_name_pattern = image_name_pattern
        self.image_count = 0

    def __enter__(self):
        if os.path.exists(self.target_dir_path):
            if self.overwrite:
                shutil.rmtree(self.target_dir_path)
                os.makedirs(self.target_dir_path)
        else:
            os.makedirs(self.target_dir_path)

        return self

    def save_image(self, image: np.ndarray, image_name: str | None = None):
        """
        Save a given image in the target directory.

        Args:
            image (np.ndarray): The image to be saved. The image must be in BGR color
                format.
            image_name (Optional[str]): The name to use for the saved image.
                If not provided, a name will be
                generated using the `image_name_pattern`.
        """
        if image_name is None:
            image_name = self.image_name_pattern.format(self.image_count)

        image_path = os.path.join(self.target_dir_path, image_name)
        cv2.imwrite(image_path, image)
        self.image_count += 1

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass
