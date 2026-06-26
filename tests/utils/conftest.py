import os

import cv2
import numpy as np
from _pytest.fixtures import fixture
from PIL import Image

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
ALL_IMAGES_LIST = [os.path.join(ASSETS_DIR, f"{i}.jpg") for i in range(1, 6)]


@fixture(scope="function")
def empty_cv2_image() -> np.ndarray:
    return np.zeros((128, 128, 3), dtype=np.uint8)


@fixture(scope="function")
def empty_pillow_image() -> Image.Image:
    return Image.new(mode="RGB", size=(128, 128), color=(0, 0, 0))


@fixture(scope="function")
def all_images() -> list[np.ndarray]:
    return [cv2.imread(path) for path in ALL_IMAGES_LIST]


@fixture(scope="function")
def one_image() -> np.ndarray:
    return cv2.imread(ALL_IMAGES_LIST[0])


@fixture(scope="function")
def two_images() -> list[np.ndarray]:
    return [cv2.imread(path) for path in ALL_IMAGES_LIST[:2]]


@fixture(scope="function")
def three_images() -> list[np.ndarray]:
    return [cv2.imread(path) for path in ALL_IMAGES_LIST[:3]]


@fixture(scope="function")
def four_images() -> list[np.ndarray]:
    return [cv2.imread(path) for path in ALL_IMAGES_LIST[:4]]


@fixture(scope="function")
def all_images_tile() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "all_images_tile.png"))


@fixture(scope="function")
def all_images_tile_and_custom_colors() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "all_images_tile_and_custom_colors.png"))


@fixture(scope="function")
def all_images_tile_and_custom_grid() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "all_images_tile_and_custom_grid.png"))


@fixture(scope="function")
def four_images_tile() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "four_images_tile.png"))


@fixture(scope="function")
def single_image_tile() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "single_image_tile.png"))


@fixture(scope="function")
def single_image_tile_enforced_grid() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "single_image_tile_enforced_grid.png"))


@fixture(scope="function")
def three_images_tile() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "three_images_tile.png"))


@fixture(scope="function")
def two_images_tile() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "two_images_tile.png"))


@fixture(scope="function")
def all_images_tile_and_custom_colors_and_titles() -> np.ndarray:
    return cv2.imread(
        os.path.join(ASSETS_DIR, "all_images_tile_and_custom_colors_and_titles.png")
    )


@fixture(scope="function")
def all_images_tile_and_titles_with_custom_configs() -> np.ndarray:
    return cv2.imread(
        os.path.join(ASSETS_DIR, "all_images_tile_and_titles_with_custom_configs.png")
    )
