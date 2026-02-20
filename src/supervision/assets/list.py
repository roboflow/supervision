from enum import Enum

BASE_VIDEO_URL = "https://media.roboflow.com/supervision/video-examples/"
BASE_IMAGE_URL = "https://media.roboflow.com/supervision/image-examples/"


class Assets(Enum):
    filename: str
    md5_hash: str

    def __new__(cls, filename: str, md5_hash: str) -> "Assets":
        obj = object.__new__(cls)
        obj._value_ = filename
        obj.filename = filename
        obj.md5_hash = md5_hash
        return obj

    @classmethod
    def list(cls) -> list[str]:
        return [asset.filename for asset in cls]


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
    | `BEACH`                | `beach-1.mp4`              | [Link](https://media.roboflow.com/supervision/video-examples/beach-1.mp4)             |
    | `BASKETBALL`           | `basketball-1.mp4`         | [Link](https://media.roboflow.com/supervision/video-examples/basketball-1.mp4)        |
    | `SKIING`               | `skiing.mp4`               | [Link](https://media.roboflow.com/supervision/video-examples/skiing.mp4)              |
    """  # noqa: E501 // docs

    VEHICLES = ("vehicles.mp4", "8155ff4e4de08cfa25f39de96483f918")
    MILK_BOTTLING_PLANT = (
        "milk-bottling-plant.mp4",
        "9e8fb6e883f842a38b3d34267290bdc7",
    )
    VEHICLES_2 = ("vehicles-2.mp4", "830af6fba21ffbf14867a7fea595937b")
    GROCERY_STORE = ("grocery-store.mp4", "48608fb4a8981f1c2469fa492adeec9c")
    SUBWAY = ("subway.mp4", "453475750691fb23c56a0cffef089194")
    MARKET_SQUARE = ("market-square.mp4", "859179bf4a21f80a8baabfdb2ed716dc")
    PEOPLE_WALKING = ("people-walking.mp4", "0574c053c8686c3f1dc0aa3743e45cb9")
    BEACH = ("beach-1.mp4", "4175d42fec4d450ed081523fd39e0cf8")
    BASKETBALL = ("basketball-1.mp4", "60d94a3c7c47d16f09d342b088012ecc")
    SKIING = ("skiing.mp4", "d30987cbab1bbc5934199cdd1b293119")


class ImageAssets(Assets):
    """
    Each member of this enum represents a image asset. The value associated with each
    member is the filename of the image.

    | Asset              | Image Filename         | Video URL                                                                             |
    |--------------------|------------------------|---------------------------------------------------------------------------------------|
    | `PEOPLE_WALKING`   | `people-walking.jpg`   | [Link](https://media.roboflow.com/supervision/image-examples/people-walking.jpg)      |
    | `SOCCER`           | `soccer.jpg`           | [Link](https://media.roboflow.com/supervision/image-examples/soccer.jpg)              |

    """  # noqa: E501 // docs

    PEOPLE_WALKING = ("people-walking.jpg", "e6bda00b47f2908eeae7df86ef995dcd")
    SOCCER = ("soccer.jpg", "0f5a4b98abf3e3973faf9e9260a7d876")


MEDIA_ASSETS: dict[str, tuple[str, str]] = {
    **{
        asset.filename: (f"{BASE_VIDEO_URL}{asset.filename}", asset.md5_hash)
        for asset in VideoAssets
    },
    **{
        asset.filename: (f"{BASE_IMAGE_URL}{asset.filename}", asset.md5_hash)
        for asset in ImageAssets
    },
}
