from __future__ import annotations

import os
import shutil
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

from supervision.draw.base import ImageType
from supervision.draw.color import Color, unify_to_bgr
from supervision.utils.conversion import (
    ensure_cv2_image_for_standalone_function,
)
from supervision.utils.internal import deprecated


@ensure_cv2_image_for_standalone_function
def crop_image(
    image: ImageType,
    xyxy: npt.NDArray[int] | list[int] | tuple[int, int, int, int],
) -> ImageType:
    """
    Crop image based on bounding box coordinates.

    Args:
        image: The image to crop.
        xyxy:
            Bounding box coordinates in `(x_min, y_min, x_max, y_max)` format.

    Returns:
        Cropped image matching input
            type.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        >>> image.shape
        (1080, 1920, 3)
        >>> xyxy = (400, 400, 800, 800)
        >>> cropped_image = sv.crop_image(image=image, xyxy=xyxy)
        >>> cropped_image.shape
        (400, 400, 3)

        ```

        ```pycon
        >>> image = np.zeros((1920, 1080), dtype=np.uint8)
        >>> image.shape
        (1920, 1080)
        >>> xyxy = (400, 400, 800, 800)
        >>> cropped_image = sv.crop_image(image=image, xyxy=xyxy)
        >>> cropped_image.shape
        (400, 400)

        ```

    ![crop-image](https://media.roboflow.com/supervision-docs/supervision-docs-crop-image-2.png){ align=center width="1000" }
    """  # noqa E501 // docs
    if isinstance(xyxy, (list, tuple)):
        xyxy = np.array(xyxy)

    xyxy = np.round(xyxy).astype(int)
    x_min, y_min, x_max, y_max = xyxy.flatten()

    if isinstance(image, np.ndarray):
        return image[y_min:y_max, x_min:x_max]

    if isinstance(image, Image.Image):
        return image.crop((x_min, y_min, x_max, y_max))

    raise TypeError(
        f"`image` must be a numpy.ndarray or PIL.Image.Image. Received {type(image)}"
    )


@ensure_cv2_image_for_standalone_function
def scale_image(image: ImageType, scale_factor: float) -> ImageType:
    """
    Scale image by given factor. Scale factor > 1.0 zooms in, < 1.0 zooms out.

    Args:
        image: The image to scale.
        scale_factor: Factor by which to scale the image.

    Returns:
        Scaled image matching input
            type.

    Raises:
        ValueError: If scale factor is non-positive.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        >>> image.shape
        (1080, 1920, 3)
        >>> scaled_image = sv.scale_image(image=image, scale_factor=0.5)
        >>> scaled_image.shape
        (540, 960, 3)

        ```

        ```pycon
        >>> image = np.zeros((1920, 1080), dtype=np.uint8)
        >>> image.shape
        (1920, 1080)
        >>> scaled_image = sv.scale_image(image=image, scale_factor=0.5)
        >>> scaled_image.shape
        (960, 540)

        ```

    ![scale-image](https://media.roboflow.com/supervision-docs/supervision-docs-scale-image-2.png){ align=center width="1000" }
    """  # noqa E501 // docs
    if scale_factor <= 0:
        raise ValueError("Scale factor must be positive.")

    width_old, height_old = image.shape[1], image.shape[0]
    width_new = int(width_old * scale_factor)
    height_new = int(height_old * scale_factor)
    return cv2.resize(image, (width_new, height_new), interpolation=cv2.INTER_LINEAR)


@ensure_cv2_image_for_standalone_function
def resize_image(
    image: ImageType,
    resolution_wh: tuple[int, int],
    keep_aspect_ratio: bool = False,
) -> ImageType:
    """
    Resize image to specified resolution. Can optionally maintain aspect ratio.

    Args:
        image: The image to resize.
        resolution_wh: Target resolution as `(width, height)`.
        keep_aspect_ratio: Flag to maintain original aspect ratio.
            Defaults to `False`.

    Returns:
        Resized image matching input
            type.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        >>> image.shape
        (1080, 1920, 3)
        >>> resized_image = sv.resize_image(
        ...     image=image, resolution_wh=(1000, 1000), keep_aspect_ratio=True
        ... )
        >>> resized_image.shape
        (562, 1000, 3)

        ```

        ```pycon
        >>> image = np.zeros((1920, 1080), dtype=np.uint8)
        >>> image.shape
        (1920, 1080)
        >>> resized_image = sv.resize_image(
        ...     image=image, resolution_wh=(1000, 1000), keep_aspect_ratio=True
        ... )
        >>> resized_image.shape
        (1000, 562)

        ```

    ![resize-image](https://media.roboflow.com/supervision-docs/supervision-docs-resize-image-2.png){ align=center width="1000" }
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


@ensure_cv2_image_for_standalone_function
def letterbox_image(
    image: ImageType,
    resolution_wh: tuple[int, int],
    color: tuple[int, int, int] | Color = Color.BLACK,
) -> ImageType:
    """
    Resize image and pad with color to achieve desired resolution while
    maintaining aspect ratio.

    Args:
        image: The image to resize and pad.
        resolution_wh: Target resolution as `(width, height)`.
        color: Padding color. If tuple, should
            be in BGR format. Defaults to `Color.BLACK`.

    Returns:
        Letterboxed image matching input
            type.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        >>> image.shape
        (1080, 1920, 3)
        >>> letterboxed_image = sv.letterbox_image(
        ...     image=image, resolution_wh=(1000, 1000)
        ... )
        >>> letterboxed_image.shape
        (1000, 1000, 3)

        ```

    ![letterbox-image](https://media.roboflow.com/supervision-docs/supervision-docs-letterbox-image-2.png){ align=center width="1000" }
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


@deprecated(
    "`overlay_image` function is deprecated and will be removed in "
    "`supervision-0.32.0`. Use `draw_image` instead."
)
def overlay_image(
    image: npt.NDArray[np.uint8],
    overlay: npt.NDArray[np.uint8],
    anchor: tuple[int, int],
) -> npt.NDArray[np.uint8]:
    """
    Overlay image onto scene at specified anchor point. Handles cases where
    overlay position is partially or completely outside scene bounds.

    Args:
        image: Background scene with shape `(height, width, 3)`.
        overlay: Image to overlay with shape
            `(height, width, 3)` or `(height, width, 4)`.
        anchor: Coordinates `(x, y)` where top-left corner
            of overlay will be placed.

    Returns:
        Scene with overlay applied, shape `(height, width, 3)`.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        >>> overlay = np.zeros((400, 400, 3), dtype=np.uint8)
        >>> overlay[:] = (0, 255, 0)  # Green overlay
        >>> result_image = sv.overlay_image(
        ...     image=image, overlay=overlay, anchor=(200, 400)
        ... )
        >>> result_image.shape
        (1000, 1000, 3)

        ```
    """
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


@ensure_cv2_image_for_standalone_function
def tint_image(
    image: ImageType,
    color: Color = Color.BLACK,
    opacity: float = 0.5,
) -> ImageType:
    """
    Tint image with solid color overlay at specified opacity.

    Args:
        image: The image to tint.
        color: Overlay tint color. Defaults to `Color.BLACK`.
        opacity: Blend ratio between overlay and image (0.0-1.0).
            Defaults to `0.5`.

    Returns:
        Tinted image matching input
            type.

    Raises:
        ValueError: If opacity is outside range [0.0, 1.0].

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> tinted_image = sv.tint_image(
        ...     image=image, color=sv.Color.ROBOFLOW, opacity=0.5
        ... )
        >>> tinted_image.shape
        (100, 100, 3)

        ```

    ![tint-image](https://media.roboflow.com/supervision-docs/supervision-docs-tint-image-2.png){ align=center width="1000" }
    """  # noqa E501 // docs
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be between 0.0 and 1.0")

    overlay = np.full_like(image, fill_value=color.as_bgr(), dtype=image.dtype)
    cv2.addWeighted(
        src1=overlay, alpha=opacity, src2=image, beta=1 - opacity, gamma=0, dst=image
    )
    return image


@ensure_cv2_image_for_standalone_function
def grayscale_image(image: ImageType) -> ImageType:
    """
    Convert image to 3-channel grayscale. Luminance channel is broadcast to
    all three channels for compatibility with color-based drawing helpers.

    Args:
        image: The image to convert to
            grayscale.

    Returns:
        3-channel grayscale image
            matching input type.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        >>> grayscale_image = sv.grayscale_image(image=image)
        >>> grayscale_image.shape
        (100, 100, 3)

        ```

    ![grayscale-image](https://media.roboflow.com/supervision-docs/supervision-docs-grayscale-image-2.png){ align=center width="1000" }
    """  # noqa E501 // docs
    grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(grayscaled, cv2.COLOR_GRAY2BGR)


def get_image_resolution_wh(image: ImageType) -> tuple[int, int]:
    """
    Get image width and height as a tuple `(width, height)` for various image formats.

    Supports both `numpy.ndarray` images (with shape `(H, W, ...)`) and
    `PIL.Image.Image` inputs.

    Args:
        image: Input image.

    Returns:
        Image resolution as `(width, height)`.

    Raises:
        ValueError: If a `numpy.ndarray` image has fewer than 2 dimensions.
        TypeError: If `image` is not a supported type (`numpy.ndarray` or
            `PIL.Image.Image`).

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        >>> sv.get_image_resolution_wh(image)
        (1920, 1080)

        ```
    """
    if isinstance(image, np.ndarray):
        if image.ndim < 2:
            raise ValueError(
                "NumPy image must have at least 2 dimensions (H, W, ...). "
                f"Received shape: {image.shape}"
            )
        height, width = image.shape[:2]
        return int(width), int(height)

    if isinstance(image, Image.Image):
        width, height = image.size
        return int(width), int(height)

    raise TypeError(
        "`image` must be a numpy.ndarray or PIL.Image.Image. "
        f"Received type: {type(image)}"
    )


class ImageSink:
    def __init__(
        self,
        target_dir_path: str,
        overwrite: bool = False,
        image_name_pattern: str = "image_{:05d}.png",
    ):
        """
        Initialize context manager for saving images to directory.

        Args:
            target_dir_path: Target directory path where images will be
                saved.
            overwrite: Whether to overwrite existing directory.
                Defaults to `False`.
            image_name_pattern: File name pattern for saved images.
                Defaults to `"image_{:05d}.png"`.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> import tempfile
            >>> import os
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     image = np.zeros((100, 100, 3), dtype=np.uint8)
            ...     with sv.ImageSink(target_dir_path=tmpdir, overwrite=True) as sink:
            ...         sink.save_image(image=image)
            ...         sink.save_image(image=image)
            ...     files = sorted(os.listdir(tmpdir))
            ...     len(files)
            2

            ```
        """
        self.target_dir_path = target_dir_path
        self.overwrite = overwrite
        self.image_name_pattern = image_name_pattern
        self.image_count = 0

    def __enter__(self) -> ImageSink:
        if os.path.exists(self.target_dir_path):
            if self.overwrite:
                shutil.rmtree(self.target_dir_path)
                os.makedirs(self.target_dir_path)
        else:
            os.makedirs(self.target_dir_path)

        return self

    def save_image(
        self, image: npt.NDArray[np.uint8], image_name: str | None = None
    ) -> None:
        """
        Save image to target directory with optional custom filename.

        Args:
            image: Image to save with shape `(height, width, 3)`
                in BGR format.
            image_name: Custom filename for saved image. If
                `None`, generates name using `image_name_pattern`. Defaults to
                `None`.
        """
        if image_name is None:
            image_name = self.image_name_pattern.format(self.image_count)

        image_path = os.path.join(self.target_dir_path, image_name)
        cv2.imwrite(image_path, image)
        self.image_count += 1

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: Any,
    ) -> None:
        pass
