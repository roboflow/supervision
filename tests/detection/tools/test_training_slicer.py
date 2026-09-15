import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.detection.tools.training_slicer import TrainingSlicer


@pytest.mark.parametrize(
    ("resolution_wh", "slice_wh", "overlap_wh", "expected_offsets"),
    [
        # Case 1: image divides evenly into slices, no overlap
        (
            (256, 256),
            (128, 128),
            (0, 0),
            np.array(
                [
                    [0, 0, 128, 128],
                    [128, 0, 256, 128],
                    [0, 128, 128, 256],
                    [128, 128, 256, 256],
                ]
            ),
        ),
        # Case 2: image smaller than a single slice
        (
            (100, 80),
            (320, 320),
            (0, 0),
            np.array([[0, 0, 100, 80]]),
        ),
        # Case 3: overlapping slices
        (
            (160, 160),
            (100, 100),
            (20, 20),
            np.array(
                [
                    [0, 0, 100, 100],
                    [60, 0, 160, 100],
                    [0, 60, 100, 160],
                    [60, 60, 160, 160],
                ]
            ),
        ),
    ],
)
def test_generate_offsets(
    resolution_wh: tuple[int, int],
    slice_wh: tuple[int, int],
    overlap_wh: tuple[int, int],
    expected_offsets: np.ndarray,
) -> None:
    offsets = TrainingSlicer._generate_offsets(
        resolution_wh=resolution_wh, slice_wh=slice_wh, overlap_wh=overlap_wh
    )

    assert np.array_equal(offsets, expected_offsets)


@pytest.mark.parametrize(
    ("slice_wh", "overlap_wh", "min_visibility", "match"),
    [
        (0, 0, 0.1, "slice_wh"),
        ((0, 10), 0, 0.1, "slice_wh"),
        (100, -1, 0.1, "overlap_wh"),
        (100, (100, 0), 0.1, "overlap_wh"),
        (100, 0, -0.1, "min_visibility"),
        (100, 0, 1.1, "min_visibility"),
    ],
)
def test_init_raises_on_invalid_arguments(
    slice_wh: int | tuple[int, int],
    overlap_wh: int | tuple[int, int],
    min_visibility: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        TrainingSlicer(
            slice_wh=slice_wh, overlap_wh=overlap_wh, min_visibility=min_visibility
        )


def test_call_splits_image_into_expected_number_of_tiles() -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    detections = Detections.empty()
    slicer = TrainingSlicer(slice_wh=100, overlap_wh=0, drop_empty_slices=False)

    result = slicer(image, detections)

    assert len(result) == 4
    for tile_image, tile_detections in result:
        assert tile_image.shape == (100, 100, 3)
        assert len(tile_detections) == 0


def test_call_drop_empty_slices_removes_tiles_without_detections() -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    detections = Detections(xyxy=np.array([[10, 10, 50, 50]], dtype=float))
    slicer = TrainingSlicer(slice_wh=100, overlap_wh=0, drop_empty_slices=True)

    result = slicer(image, detections)

    assert len(result) == 1
    tile_image, tile_detections = result[0]
    assert tile_image.shape == (100, 100, 3)
    np.testing.assert_array_equal(tile_detections.xyxy, [[10, 10, 50, 50]])


def test_call_localizes_box_fully_inside_one_tile() -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    detections = Detections(
        xyxy=np.array([[110, 120, 140, 160]], dtype=float),
        class_id=np.array([7]),
    )
    slicer = TrainingSlicer(slice_wh=100, overlap_wh=0, drop_empty_slices=True)

    result = slicer(image, detections)

    assert len(result) == 1
    _, tile_detections = result[0]
    np.testing.assert_array_equal(tile_detections.xyxy, [[10, 20, 40, 60]])
    np.testing.assert_array_equal(tile_detections.class_id, [7])


def test_call_drops_box_below_min_visibility_at_tile_boundary() -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    # 60x60 box straddling the (100, 100) tile boundary: only a 10x10 corner
    # (100 / 3600 ~= 2.8%) falls inside the top-left tile.
    detections = Detections(xyxy=np.array([[90, 90, 150, 150]], dtype=float))
    slicer = TrainingSlicer(
        slice_wh=100, overlap_wh=0, min_visibility=0.5, drop_empty_slices=True
    )

    result = slicer(image, detections)

    # Only the bottom-right tile keeps the box: it sees a 50x50 (~69%) crop.
    assert len(result) == 1
    _, tile_detections = result[0]
    np.testing.assert_array_equal(tile_detections.xyxy, [[0, 0, 50, 50]])


def test_call_keeps_and_clips_box_at_min_visibility_zero() -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    detections = Detections(xyxy=np.array([[90, 90, 150, 150]], dtype=float))
    slicer = TrainingSlicer(
        slice_wh=100, overlap_wh=0, min_visibility=0.0, drop_empty_slices=True
    )

    result = slicer(image, detections)

    # Every tile that touches the box keeps a clipped copy of it.
    assert len(result) == 4
    for _, tile_detections in result:
        assert len(tile_detections) == 1
        xyxy = tile_detections.xyxy[0]
        assert (xyxy >= 0).all()
        assert xyxy[0] <= xyxy[2] <= 100
        assert xyxy[1] <= xyxy[3] <= 100


def test_call_slices_masks_to_local_tile_coordinates() -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    mask = np.zeros((1, 200, 200), dtype=bool)
    mask[0, 90:150, 90:150] = True
    detections = Detections(xyxy=np.array([[90, 90, 150, 150]], dtype=float), mask=mask)
    slicer = TrainingSlicer(
        slice_wh=100, overlap_wh=0, min_visibility=0.5, drop_empty_slices=True
    )

    result = slicer(image, detections)

    assert len(result) == 1
    _, tile_detections = result[0]
    assert tile_detections.mask.shape == (1, 100, 100)
    # Only the (100, 100)-(150, 150) portion of the mask falls in this tile,
    # localized to (0, 0)-(50, 50).
    assert tile_detections.mask[0].sum() == 50 * 50
    assert tile_detections.mask[0, :50, :50].all()
    assert not tile_detections.mask[0, 50:, :].any()
    assert not tile_detections.mask[0, :, 50:].any()


def test_call_returns_tiles_in_row_major_order() -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    detections = Detections.empty()
    slicer = TrainingSlicer(slice_wh=100, overlap_wh=0, drop_empty_slices=False)

    offsets = TrainingSlicer._generate_offsets(
        resolution_wh=(200, 200), slice_wh=(100, 100), overlap_wh=(0, 0)
    )
    result = slicer(image, detections)

    assert len(result) == len(offsets)
