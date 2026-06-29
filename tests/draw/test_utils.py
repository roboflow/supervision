import cv2
import numpy as np
import pytest

from supervision.draw.utils import draw_image
from supervision.geometry.core import Rect


def test_draw_image_invalid_path_raises_oserror(tmp_path) -> None:
    """Existing but undecodable image files raise OSError."""
    invalid_image_path = tmp_path / "invalid_image.dat"
    invalid_image_path.write_bytes(b"not an image")
    scene = np.zeros((100, 100, 3), dtype=np.uint8)
    rect = Rect(x=0, y=0, width=100, height=100)

    with pytest.raises(OSError, match="Could not decode image path"):
        draw_image(
            scene=scene,
            image=str(invalid_image_path),
            opacity=1.0,
            rect=rect,
        )


def test_draw_image_valid_image(tmp_path) -> None:
    """Valid image files are decoded and drawn onto the scene."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), image)

    scene = np.zeros((100, 100, 3), dtype=np.uint8)
    rect = Rect(x=0, y=0, width=100, height=100)

    result = draw_image(
        scene=scene,
        image=str(image_path),
        opacity=1.0,
        rect=rect,
    )

    assert isinstance(result, np.ndarray)


def test_draw_image_grayscale_file_raises_value_error(tmp_path) -> None:
    """Grayscale image files raise ValueError before channel access."""
    image = np.zeros((100, 100), dtype=np.uint8)
    image_path = tmp_path / "grayscale.png"
    cv2.imwrite(str(image_path), image)

    scene = np.zeros((100, 100, 3), dtype=np.uint8)
    rect = Rect(x=0, y=0, width=100, height=100)

    with pytest.raises(ValueError, match="3 or 4 channels"):
        draw_image(
            scene=scene,
            image=str(image_path),
            opacity=1.0,
            rect=rect,
        )


def test_draw_image_grayscale_array_raises_value_error() -> None:
    """Grayscale image arrays raise ValueError before channel access."""
    image = np.zeros((100, 100), dtype=np.uint8)
    scene = np.zeros((100, 100, 3), dtype=np.uint8)
    rect = Rect(x=0, y=0, width=100, height=100)

    with pytest.raises(ValueError, match="3 or 4 channels"):
        draw_image(
            scene=scene,
            image=image,
            opacity=1.0,
            rect=rect,
        )
