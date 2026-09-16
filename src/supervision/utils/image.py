from __future__ import annotations

import os
import shutil
import tempfile
import urllib.parse
from hashlib import md5
from pathlib import Path
from types import TracebackType
from typing import cast

import numpy as np
import numpy.typing as npt
from PIL import Image

from supervision import _cv2 as cv2
from supervision.draw.base import ImageType
from supervision.draw.color import Color, unify_to_bgr
from supervision.utils.conversion import (
    ensure_cv2_image_for_standalone_function,
)
from supervision.utils.file import (
    SUPERVISION_CACHE_DIR,
    _download_to_file,
    _normalize_http_url,
)

DEFAULT_IMAGE_URL_CACHE_DIR = SUPERVISION_CACHE_DIR / "image-url"


def _get_image_url_cache_path(value: str, cache_dir: str | Path | None) -> Path:
    """Build the cache file path for a URL: `<cache root>/<md5(url)><suffix>`."""
    cache_root = (
        DEFAULT_IMAGE_URL_CACHE_DIR
        if cache_dir is None
        else Path(cache_dir).expanduser().resolve()
    )
    url_path = urllib.parse.urlparse(value).path
    suffix = Path(url_path).suffix or ".image"
    url_hash = md5(value.encode("utf-8"), usedforsecurity=False).hexdigest()
    return cache_root / f"{url_hash}{suffix}"


def _decode_image_from_bytes(
    value: bytes,
    cv_imread_flags: int,
) -> npt.NDArray[np.uint8]:
    """Decode raw image bytes into an OpenCV image, raising on undecodable data."""
    image = cv2.imdecode(
        np.frombuffer(value, dtype=np.uint8),
        cv_imread_flags,
    )
    if image is None:
        raise ValueError("Data pointed by URL could not be decoded into image.")

    return cast(npt.NDArray[np.uint8], image)


def load_image_from_url(
    value: str,
    cv_imread_flags: int = cv2.IMREAD_COLOR,
    timeout: float = 30.0,
    use_cache: bool = True,
    cache_dir: str | Path | None = None,
    force_reload: bool = False,
) -> npt.NDArray[np.uint8]:
    """Load an image from a URL as an OpenCV image.

    Args:
        value: HTTP(S) URL of the image.
        cv_imread_flags: OpenCV image read flag passed to `cv2.imdecode`.
            Defaults to `cv2.IMREAD_COLOR`.
        timeout: Request timeout in seconds. Defaults to `30.0`.
        use_cache: If `True`, cache downloaded image bytes locally and reuse them
            on repeated calls. Defaults to `True`.
        cache_dir: Directory where downloaded image bytes are cached. If `None`,
            uses the system temporary directory. Defaults to `None`.
        force_reload: If `True`, re-download the image and refresh the cache.
            Defaults to `False`.

    Returns:
        Image as a NumPy array in the format selected by `cv_imread_flags`.

    Raises:
        ValueError: If the URL is invalid or the downloaded bytes cannot be decoded.
        requests.RequestException: If the request fails or returns an error status.

    Examples:
        ```python
        import supervision as sv

        image = sv.load_image_from_url(
            "https://media.roboflow.com/quickstart/dog.jpeg"
        )
        image.shape

        ```
    """
    prepared_url = _normalize_http_url(url=value)
    if not use_cache:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_target = Path(temp_dir) / "image"
            _download_to_file(prepared_url, temp_target, timeout=timeout)
            return _decode_image_from_bytes(
                value=temp_target.read_bytes(),
                cv_imread_flags=cv_imread_flags,
            )

    cache_path = _get_image_url_cache_path(
        value=prepared_url,
        cache_dir=cache_dir,
    )
    if cache_path.exists() and not force_reload:
        try:
            return _decode_image_from_bytes(
                value=cache_path.read_bytes(),
                cv_imread_flags=cv_imread_flags,
            )
        except ValueError:
            cache_path.unlink(missing_ok=True)

    _download_to_file(prepared_url, cache_path, timeout=timeout)
    try:
        return _decode_image_from_bytes(
            value=cache_path.read_bytes(),
            cv_imread_flags=cv_imread_flags,
        )
    except ValueError:
        cache_path.unlink(missing_ok=True)
        raise


@ensure_cv2_image_for_standalone_function
def crop_image(
    image: ImageType,
    xyxy: npt.NDArray[np.number] | list[int] | tuple[int, int, int, int],
) -> ImageType:
    """Crop image based on bounding box coordinates.

    Args:
        image: The image to crop.
        xyxy:
            Bounding box coordinates in `(x_min, y_min, x_max, y_max)` format.

    Returns:
        Cropped image matching input
            type.

    Note:
        Coordinates are rounded to integers and clipped to the image bounds
        before slicing. This keeps NumPy and Pillow inputs aligned and avoids
        negative-index wrap-around on NumPy arrays.

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
    xyxy_arr = np.asarray(xyxy, dtype=np.float64).round().astype(np.int32)
    x_min, y_min, x_max, y_max = xyxy_arr.flatten()

    if isinstance(image, np.ndarray):
        height, width = image.shape[:2]
        x_min = int(np.clip(x_min, 0, width))
        y_min = int(np.clip(y_min, 0, height))
        x_max = int(np.clip(x_max, 0, width))
        y_max = int(np.clip(y_max, 0, height))
        return image[y_min:y_max, x_min:x_max]

    if isinstance(image, Image.Image):
        width, height = image.size
        x_min = int(np.clip(x_min, 0, width))
        y_min = int(np.clip(y_min, 0, height))
        x_max = int(np.clip(x_max, 0, width))
        y_max = int(np.clip(y_max, 0, height))
        return image.crop((float(x_min), float(y_min), float(x_max), float(y_max)))

    raise TypeError(
        f"`image` must be a numpy.ndarray or PIL.Image.Image. Received {type(image)}"
    )


@ensure_cv2_image_for_standalone_function
def scale_image(image: ImageType, scale_factor: float) -> ImageType:
    """Scale image by given factor. Scale factor > 1.0 zooms in, < 1.0 zooms out.

    Args:
        image: The image to scale.
        scale_factor: Factor by which to scale the image.

    Returns:
        Scaled image matching input
            type.

    Raises:
        TypeError: If `image` is not a `numpy.ndarray` or `PIL.Image.Image`.
        ValueError: If scale factor is non-positive.

    Note:
        Each axis keeps at least one pixel, so a factor small enough to round an
        axis of a small image down to zero returns a one-pixel-wide or
        one-pixel-tall image rather than raising.

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
    # A small factor on a small image truncates an axis to zero, and `cv2.resize`
    # answers a zero-sized target with a bare assertion. One pixel is the smallest
    # image that still exists, so clamp there instead.
    width_new = max(1, int(width_old * scale_factor))
    height_new = max(1, int(height_old * scale_factor))
    return cast(
        npt.NDArray[np.uint8],
        cv2.resize(image, (width_new, height_new), interpolation=cv2.INTER_LINEAR),
    )


@ensure_cv2_image_for_standalone_function
def resize_image(
    image: ImageType,
    resolution_wh: tuple[int, int],
    keep_aspect_ratio: bool = False,
) -> ImageType:
    """Resize image to specified resolution. Can optionally maintain aspect ratio.

    Args:
        image: The image to resize.
        resolution_wh: Target resolution as `(width, height)`.
        keep_aspect_ratio: Flag to maintain original aspect ratio.
            Defaults to `False`.

    Returns:
        Resized image matching input
            type.

    Raises:
        TypeError: If `image` is not a `numpy.ndarray` or `PIL.Image.Image`.

    Note:
        With `keep_aspect_ratio=True` the fitted axis keeps at least one pixel, so
        an aspect ratio too extreme to fit the target box returns a one-pixel-wide
        or one-pixel-tall image rather than raising.

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
        # The fitted axis truncates to zero once the aspect ratio outgrows the target
        # box, and `cv2.resize` answers a zero-sized target with a bare assertion.
        # One pixel is the smallest image that still exists, so clamp there instead.
        if image_ratio >= target_ratio:
            width_new = resolution_wh[0]
            height_new = max(1, int(resolution_wh[0] / image_ratio))
        else:
            height_new = resolution_wh[1]
            width_new = max(1, int(resolution_wh[1] * image_ratio))
    else:
        width_new, height_new = resolution_wh

    return cast(
        npt.NDArray[np.uint8],
        cv2.resize(image, (width_new, height_new), interpolation=cv2.INTER_LINEAR),
    )


@ensure_cv2_image_for_standalone_function
def letterbox_image(
    image: ImageType,
    resolution_wh: tuple[int, int],
    color: tuple[int, int, int] | Color = Color.BLACK,
) -> ImageType:
    """Resize image and pad with color to achieve desired resolution while maintaining
    aspect ratio.

    Args:
        image: The image to resize and pad. Accepts BGR arrays of shape
            ``(H, W, 3)``, BGRA arrays of shape ``(H, W, 4)``, grayscale
            arrays of shape ``(H, W)``, or a PIL ``Image``.
        resolution_wh: Target resolution as `(width, height)`.
        color: Padding color. If tuple, should be in BGR format.
            Defaults to `Color.BLACK`.

    Returns:
        Letterboxed image matching input type.

    Raises:
        TypeError: If `image` is not a `numpy.ndarray` or `PIL.Image.Image`.

    Note:
        For BGRA inputs, the alpha channel in the padding region is set to
        0 (fully transparent). Grayscale inputs receive scalar padding
        from ``color[0]``.

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
        >>> gray = np.zeros((4, 6), dtype=np.uint8)
        >>> sv.letterbox_image(image=gray, resolution_wh=(10, 10)).shape
        (10, 10)

        ```

    ![letterbox-image](https://media.roboflow.com/supervision-docs/supervision-docs-letterbox-image-2.png){ align=center width="1000" }
    """  # noqa E501 // docs
    color = unify_to_bgr(color=color)
    resized_image = resize_image(
        image=image, resolution_wh=resolution_wh, keep_aspect_ratio=True
    )
    height_new, width_new = resized_image.shape[:2]
    padding_top = (resolution_wh[1] - height_new) // 2
    padding_bottom = resolution_wh[1] - height_new - padding_top
    padding_left = (resolution_wh[0] - width_new) // 2
    padding_right = resolution_wh[0] - width_new - padding_left
    image_with_borders = cast(
        npt.NDArray[np.uint8],
        cv2.copyMakeBorder(
            resized_image,
            padding_top,
            padding_bottom,
            padding_left,
            padding_right,
            cv2.BORDER_CONSTANT,
            value=color,
        ),
    )

    return image_with_borders


def _overlay_image(
    image: npt.NDArray[np.uint8],
    overlay: npt.NDArray[np.uint8],
    anchor: tuple[int, int],
) -> npt.NDArray[np.uint8]:
    """Overlay `overlay` onto `image` at `anchor`, clipping to scene bounds.

    Args:
        image: Background BGR array of shape ``(H, W, 3)``. Modified in place
            and returned.
        overlay: Overlay array of shape ``(H, W, 3)`` or ``(H, W, 4)``; channel
            4, when present, is treated as alpha.
        anchor: ``(x, y)`` pixel position of the overlay top-left corner. May
            be negative (partial off-screen placement is clipped).

    Returns:
        The ``image`` array with the overlay applied.
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
        alpha_f32 = alpha[:, :, None].astype(np.float32) / 255.0
        overlay_color = cv2.merge((b, g, r)).astype(np.float32)

        roi = image[y_min:y_max, x_min:x_max].astype(np.float32)
        blended = roi * (1 - alpha_f32) + overlay_color * alpha_f32
        image[y_min:y_max, x_min:x_max] = np.clip(blended, 0, 255).astype(np.uint8)
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
    """Tint image with solid color overlay at specified opacity.

    Args:
        image: The image to tint.
        color: Overlay tint color. Defaults to `Color.BLACK`.
        opacity: Blend ratio between overlay and image (0.0-1.0).
            Defaults to `0.5`.

    Returns:
        Tinted image matching input
            type. The input image is left unchanged.

    Raises:
        TypeError: If `image` is not a `numpy.ndarray` or `PIL.Image.Image`.
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
        >>> int(image.max())
        0

        ```

    ![tint-image](https://media.roboflow.com/supervision-docs/supervision-docs-tint-image-2.png){ align=center width="1000" }
    """  # noqa E501 // docs
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be between 0.0 and 1.0")

    overlay = np.full_like(image, fill_value=color.as_bgr(), dtype=image.dtype)
    # No `dst`: let the blend allocate its own buffer. Passing `image` there wrote
    # the tint back into the caller's array for a NumPy input, while a Pillow input
    # was shielded by the ndarray conversion the decorator makes.
    return cast(
        npt.NDArray[np.uint8],
        cv2.addWeighted(
            src1=overlay, alpha=opacity, src2=image, beta=1 - opacity, gamma=0
        ),
    )


@ensure_cv2_image_for_standalone_function
def grayscale_image(image: ImageType) -> ImageType:
    """Convert image to 3-channel grayscale. Luminance channel is broadcast to all three
    channels for compatibility with color-based drawing helpers.

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
    assert isinstance(image, np.ndarray)
    grayscaled = cast(npt.NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return cast(npt.NDArray[np.uint8], cv2.cvtColor(grayscaled, cv2.COLOR_GRAY2BGR))


def get_image_resolution_wh(image: ImageType) -> tuple[int, int]:
    """Get image width and height as a tuple `(width, height)` for various image
    formats.

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
    """Save sequential images into a directory through a context manager.

    `ImageSink` creates the target directory on entry and writes each image using
    `save_image`, incrementing the image name pattern after every save.
    """

    def __init__(
        self,
        target_dir_path: str,
        overwrite: bool = False,
        image_name_pattern: str = "image_{:05d}.png",
    ) -> None:
        """Initialize context manager for saving images to directory.

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
        """Save image to target directory with optional custom filename.

        Args:
            image: Image to save with shape `(height, width, 3)`
                in BGR format.
            image_name: Custom filename for saved image. If
                `None`, generates name using `image_name_pattern`. Defaults to
                `None`.

        Raises:
            OSError: If `cv2.imwrite` cannot write the image to disk.
        """
        if image_name is None:
            image_name = self.image_name_pattern.format(self.image_count)

        image_path = os.path.join(self.target_dir_path, image_name)
        if not cv2.imwrite(image_path, image):
            raise OSError(f"Failed to save image to path: {image_path}")
        self.image_count += 1

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None:
        pass
