from supervision.assets.list import BASE_VIDEO_URL, VIDEO_ASSETS, VideoAssets


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


def test_video_assets_enum_values():
    """Test that VideoAssets enum members have correct values."""
    assert VideoAssets.VEHICLES.value == "vehicles.mp4"
    assert VideoAssets.MILK_BOTTLING_PLANT.value == "milk-bottling-plant.mp4"
    assert VideoAssets.VEHICLES_2.value == "vehicles-2.mp4"
    assert VideoAssets.GROCERY_STORE.value == "grocery-store.mp4"
    assert VideoAssets.SUBWAY.value == "subway.mp4"
    assert VideoAssets.MARKET_SQUARE.value == "market-square.mp4"
    assert VideoAssets.PEOPLE_WALKING.value == "people-walking.mp4"
    assert VideoAssets.BEACH.value == "beach-1.mp4"
    assert VideoAssets.BASKETBALL.value == "basketball-1.mp4"
    assert VideoAssets.SKIING.value == "skiing.mp4"


def test_video_assets_dict_keys():
    """Test that VIDEO_ASSETS has all VideoAssets as keys."""
    expected_keys = {asset.value for asset in VideoAssets}
    assert set(VIDEO_ASSETS.keys()) == expected_keys


def test_video_assets_dict_values():
    """Test that VIDEO_ASSETS values are tuples of (url, md5_hash)."""
    for filename, (url, md5_hash) in VIDEO_ASSETS.items():
        assert isinstance(url, str)
        assert url.startswith(BASE_VIDEO_URL)
        assert url.endswith(filename)
        assert isinstance(md5_hash, str)
        assert len(md5_hash) == 32  # MD5 hash length
