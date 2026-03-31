from supervision.assets.list import (
    BASE_IMAGE_URL,
    BASE_VIDEO_URL,
    MEDIA_ASSETS,
    ImageAssets,
    VideoAssets,
)


def test_video_assets_list():
    """Test that VideoAssets.list() returns all video filenames."""
    expected_filenames = [
        "vehicles.mp4",
        "milk-bottling-plant.mp4",
        "vehicles-2.mp4",
        "grocery-store.mp4",
        "subway.mp4",
        "market-square.mp4",
        "people-walking.mp4",
        "beach-1.mp4",
        "basketball-1.mp4",
        "skiing.mp4",
    ]
    assert VideoAssets.list() == expected_filenames


def test_image_assets_list():
    """Test that ImageAssets.list() returns all image filenames."""
    expected_filenames = [
        "people-walking.jpg",
        "soccer.jpg",
    ]
    assert ImageAssets.list() == expected_filenames


def test_video_assets_values():
    """Test that VideoAssets enum members have correct attributes."""
    assert VideoAssets.VEHICLES.filename == "vehicles.mp4"
    assert VideoAssets.VEHICLES.md5_hash == "8155ff4e4de08cfa25f39de96483f918"
    assert VideoAssets.VEHICLES.value == "vehicles.mp4"


def test_image_assets_values():
    """Test that ImageAssets enum members have correct attributes."""
    assert ImageAssets.SOCCER.filename == "soccer.jpg"
    assert ImageAssets.SOCCER.md5_hash == "0f5a4b98abf3e3973faf9e9260a7d876"
    assert ImageAssets.SOCCER.value == "soccer.jpg"


def test_media_assets_dict_keys():
    """Test that MEDIA_ASSETS has all VideoAssets and ImageAssets as keys."""
    expected_keys = {asset.filename for asset in VideoAssets} | {
        asset.filename for asset in ImageAssets
    }
    assert set(MEDIA_ASSETS.keys()) == expected_keys


def test_media_assets_dict_values():
    """Test that MEDIA_ASSETS values are tuples of (url, md5_hash)."""
    for filename, (url, md5_hash) in MEDIA_ASSETS.items():
        assert isinstance(url, str)
        if filename.endswith(".mp4"):
            assert url.startswith(BASE_VIDEO_URL)
        elif filename.endswith(".jpg"):
            assert url.startswith(BASE_IMAGE_URL)
        assert url.endswith(filename)
        assert isinstance(md5_hash, str)
        assert len(md5_hash) == 32  # MD5 hash length
