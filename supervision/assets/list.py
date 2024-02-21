from enum import Enum
from typing import Dict, Tuple

BASE_VIDEO_URL = "https://media.roboflow.com/supervision/video-examples/"
BASE_IMAGE_URL = "https://media.roboflow.com/inference/"


class Assets(Enum):
    def __init__(self, filename: str, hash: str):
        self.filename = filename
        self.hash = hash


class VideoAssets(Assets):
    """
    Each member of this class represents a video asset. The value associated with each
    member has a filename and hash of the video. File names and links can be seen below.

    | Asset                  | Video Filename             | Video URL                                                                             |
    |------------------------|----------------------------|---------------------------------------------------------------------------------------|
    | `VEHICLES`             | `vehicles.mp4`             | [Link](https://media.roboflow.com/supervision/video-examples/vehicles.mp4)            |
    | `MILK_BOTTLING_PLANT`  | `milk-bottling-plant.mp4`  | [Link](https://media.roboflow.com/supervision/video-examples/milk-bottling-plant.mp4) |
    | `VEHICLES_2`           | `vehicles-2.mp4`           | [Link](https://media.roboflow.com/supervision/video-examples/vehicles-2.mp4)          |
    | `GROCERY_STORE`        | `grocery-store.mp4`        | [Link](https://media.roboflow.com/supervision/video-examples/grocery-store.mp4)       |
    | `SUBWAY`               | `subway.mp4`               | [Link](https://media.roboflow.com/supervision/video-examples/subway.mp4)              |
    | `MARKET_SQUARE`        | `market-square.mp4`        | [Link](https://media.roboflow.com/supervision/video-examples/market-square.mp4)       |
    | `PEOPLE_WALKING`       | `people-walking.mp4`       | [Link](https://media.roboflow.com/supervision/video-examples/people-walking.mp4)      |
    """  # noqa: E501 // docs

    VEHICLES = ("vehicles.mp4", "8155ff4e4de08cfa25f39de96483f918")
    MILK_BOTTLING_PLANT = (
        "milk-bottling-plant.mp4",
        "9e8fb6e883f842a38b3d34267290bdc7",
    )
    VEHICLES_2 = ("vehicles-2.mp4", "830af6fba21ffbf14867a7fea595937b")
    GROCERY_STORE = ("grocery-store.mp4", "453475750691fb23c56a0cffef089194")
    SUBWAY = ("subway.mp4", "453475750691fb23c56a0cffef089194")
    MARKET_SQUARE = ("market-square.mp4", "859179bf4a21f80a8baabfdb2ed716dc")
    PEOPLE_WALKING = ("people-walking.mp4", "0574c053c8686c3f1dc0aa3743e45cb9")


class ImageAssets(Assets):
    """
    Each member of this enum represents a image asset. The value associated with each
    member is the filename of the image.

    | Asset                  | Image Filename             | Video URL                                                                             |
    |------------------------|----------------------------|---------------------------------------------------------------------------------------|
    | `PEOPLE_WALKING`       | `people-walking.jpg`       | [Link](https://media.roboflow.com/inference/people-walking.jpg)                      |

    """  # noqa: E501 // docs

    PEOPLE_WALKING = ("people-walking.jpg", "e6bda00b47f2908eeae7df86ef995dcd")


ASSETS: Dict[str, Tuple[str, str]] = {
    **{
        asset.value[0]: (f"{BASE_VIDEO_URL}{asset.value[0]}", asset.value[1])
        for asset in VideoAssets
    },
    **{
        asset.value[0]: (f"{BASE_IMAGE_URL}{asset.value[0]}", asset.value[1])
        for asset in ImageAssets
    },
}
