import numpy as np
from _pytest.fixtures import fixture
from PIL import Image


@fixture(scope="function")
def empty_opencv_image() -> np.ndarray:
    return np.zeros((128, 128, 3), dtype=np.uint8)


@fixture(scope="function")
def empty_pillow_image() -> Image.Image:
    return Image.new(mode="RGB", size=(128, 128), color=(0, 0, 0))
