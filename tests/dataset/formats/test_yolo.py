from __future__ import annotations

import os
import tempfile
from contextlib import ExitStack as DoesNotRaise
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from supervision.dataset.formats.yolo import (
    _image_name_to_annotation_name,
    _with_seg_mask,
    detections_to_yolo_annotations,
    load_yolo_annotations,
    object_to_yolo,
    yolo_annotations_to_detections,
)
from supervision.detection.core import Detections


def _mock_simple_mask(resolution_wh: tuple[int, int], box: list[int]) -> np.ndarray:
    x_min, y_min, x_max, y_max = box
    mask = np.full(resolution_wh, False, dtype=bool)
    mask[y_min:y_max, x_min:x_max] = True
    return mask


# The result of _mock_simple_mask is a little different from the result produced by cv2.
def _arrays_almost_equal(
    arr1: np.ndarray, arr2: np.ndarray, threshold: float = 0.99
) -> bool:
    equal_elements = np.equal(arr1, arr2)
    proportion_equal = np.mean(equal_elements)
    return proportion_equal >= threshold


@pytest.mark.parametrize(
    ("lines", "expected_result", "exception"),
    [
        ([], False, DoesNotRaise()),  # empty yolo annotation file
        (
            ["0 0.5 0.5 0.2 0.2"],
            False,
            DoesNotRaise(),
        ),  # yolo annotation file with single line with box
        (
            ["0 0.50 0.50 0.20 0.20", "1 0.11 0.47 0.22 0.30"],
            False,
            DoesNotRaise(),
        ),  # yolo annotation file with two lines with box
        (
            ["0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6"],
            True,
            DoesNotRaise(),
        ),  # yolo annotation file with single line with polygon
        (
            ["0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6", "1 0.11 0.47 0.22 0.30"],
            True,
            DoesNotRaise(),
        ),  # yolo annotation file with two lines - one box and one polygon
    ],
)
def test_with_mask(
    lines: list[str], expected_result: bool | None, exception: Exception
) -> None:
    with exception:
        result = _with_seg_mask(lines=lines)
        assert result == expected_result


@pytest.mark.parametrize(
    ("lines", "resolution_wh", "with_masks", "expected_result", "exception"),
    [
        (
            [],
            (1000, 1000),
            False,
            Detections.empty(),
            DoesNotRaise(),
        ),  # empty yolo annotation file
        (
            ["0 0.5 0.5 0.2 0.2"],
            (1000, 1000),
            False,
            Detections(
                xyxy=np.array([[400, 400, 600, 600]], dtype=np.float32),
                class_id=np.array([0], dtype=int),
            ),
            DoesNotRaise(),
        ),  # yolo annotation file with single line with box
        (
            ["0 0.50 0.50 0.20 0.20", "1 0.11 0.47 0.22 0.30"],
            (1000, 1000),
            False,
            Detections(
                xyxy=np.array(
                    [[400, 400, 600, 600], [0, 320, 220, 620]], dtype=np.float32
                ),
                class_id=np.array([0, 1], dtype=int),
            ),
            DoesNotRaise(),
        ),  # yolo annotation file with two lines with box
        (
            ["0 0.5 0.5 0.2 0.2"],
            (1000, 1000),
            True,
            Detections(
                xyxy=np.array([[400, 400, 600, 600]], dtype=np.float32),
                class_id=np.array([0], dtype=int),
                mask=np.array(
                    [
                        _mock_simple_mask(
                            resolution_wh=(1000, 1000), box=[400, 400, 600, 600]
                        )
                    ],
                    dtype=bool,
                ),
            ),
            DoesNotRaise(),
        ),  # yolo annotation file with single line with box in with_masks mode
        (
            ["0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6"],
            (1000, 1000),
            True,
            Detections(
                xyxy=np.array([[400, 400, 600, 600]], dtype=np.float32),
                class_id=np.array([0], dtype=int),
                mask=np.array(
                    [
                        _mock_simple_mask(
                            resolution_wh=(1000, 1000), box=[400, 400, 600, 600]
                        )
                    ],
                    dtype=bool,
                ),
            ),
            DoesNotRaise(),
        ),  # yolo annotation file with single line with polygon
        (
            ["0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6", "1 0.11 0.47 0.22 0.30"],
            (1000, 1000),
            True,
            Detections(
                xyxy=np.array(
                    [[400, 400, 600, 600], [0, 320, 220, 620]], dtype=np.float32
                ),
                class_id=np.array([0, 1], dtype=int),
                mask=np.array(
                    [
                        _mock_simple_mask(
                            resolution_wh=(1000, 1000), box=[400, 400, 600, 600]
                        ),
                        _mock_simple_mask(
                            resolution_wh=(1000, 1000), box=[0, 320, 220, 620]
                        ),
                    ],
                    dtype=bool,
                ),
            ),
            DoesNotRaise(),
        ),  # yolo annotation file with two lines -
        # one box and one polygon in with_masks mode
        (
            ["0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6", "1 0.11 0.47 0.22 0.30"],
            (1000, 1000),
            False,
            Detections(
                xyxy=np.array(
                    [[400, 400, 600, 600], [0, 320, 220, 620]], dtype=np.float32
                ),
                class_id=np.array([0, 1], dtype=int),
            ),
            DoesNotRaise(),
        ),  # yolo annotation file with two lines - one box and one polygon
        (
            ["0 0.4056 0.4078 0.5967 0.4089 0.5978 0.6012 0.4067 0.5989"],
            (1000, 1000),
            True,
            Detections(
                xyxy=np.array([[405.6, 407.8, 597.8, 601.2]], dtype=np.float32),
                class_id=np.array([0], dtype=int),
                mask=np.array(
                    [
                        _mock_simple_mask(
                            resolution_wh=(1000, 1000), box=[406, 408, 598, 601]
                        )
                    ],
                    dtype=bool,
                ),
            ),
            DoesNotRaise(),
        ),
    ],
)
def test_yolo_annotations_to_detections(
    lines: list[str],
    resolution_wh: tuple[int, int],
    with_masks: bool,
    expected_result: Detections | None,
    exception: Exception,
) -> None:
    with exception:
        result = yolo_annotations_to_detections(
            lines=lines, resolution_wh=resolution_wh, with_masks=with_masks
        )
        assert np.array_equal(result.xyxy, expected_result.xyxy)
        assert np.array_equal(result.class_id, expected_result.class_id)
        assert (
            result.mask is None and expected_result.mask is None
        ) or _arrays_almost_equal(result.mask, expected_result.mask)


@pytest.mark.parametrize(
    ("image_name", "expected_result", "exception"),
    [
        ("image.png", "image.txt", DoesNotRaise()),  # simple png image
        ("image.jpeg", "image.txt", DoesNotRaise()),  # simple jpeg image
        ("image.jpg", "image.txt", DoesNotRaise()),  # simple jpg image
        (
            "image.000.jpg",
            "image.000.txt",
            DoesNotRaise(),
        ),  # jpg image with multiple dots in name
    ],
)
def test_image_name_to_annotation_name(
    image_name: str, expected_result: str | None, exception: Exception
) -> None:
    with exception:
        result = _image_name_to_annotation_name(image_name=image_name)
        assert result == expected_result


@pytest.mark.parametrize(
    ("xyxy", "class_id", "image_shape", "polygon", "expected_result", "exception"),
    [
        (
            np.array([100, 100, 200, 200], dtype=np.float32),
            1,
            (1000, 1000, 3),
            None,
            "1 0.15000 0.15000 0.10000 0.10000",
            DoesNotRaise(),
        ),  # square bounding box on square image
        (
            np.array([100, 100, 200, 200], dtype=np.float32),
            1,
            (800, 1000, 3),
            None,
            "1 0.15000 0.18750 0.10000 0.12500",
            DoesNotRaise(),
        ),  # square bounding box on horizontal image
        (
            np.array([100, 100, 200, 200], dtype=np.float32),
            1,
            (1000, 800, 3),
            None,
            "1 0.18750 0.15000 0.12500 0.10000",
            DoesNotRaise(),
        ),  # square bounding box on vertical image
        (
            np.array([100, 200, 200, 400], dtype=np.float32),
            1,
            (1000, 1000, 3),
            None,
            "1 0.15000 0.30000 0.10000 0.20000",
            DoesNotRaise(),
        ),  # horizontal bounding box on square image
        (
            np.array([200, 100, 400, 200], dtype=np.float32),
            1,
            (1000, 1000, 3),
            None,
            "1 0.30000 0.15000 0.20000 0.10000",
            DoesNotRaise(),
        ),  # vertical bounding box on square image
        (
            np.array([100, 100, 200, 200], dtype=np.float32),
            1,
            (1000, 1000, 3),
            np.array(
                [[100, 100], [200, 100], [200, 200], [100, 100]], dtype=np.float32
            ),
            "1 0.10000 0.10000 0.20000 0.10000 0.20000 0.20000 0.10000 0.10000",
            DoesNotRaise(),
        ),  # square mask on square image
    ],
)
def test_object_to_yolo(
    xyxy: np.ndarray,
    class_id: int,
    image_shape: tuple[int, int, int],
    polygon: np.ndarray | None,
    expected_result: str | None,
    exception: Exception,
) -> None:
    with exception:
        result = object_to_yolo(
            xyxy=xyxy, class_id=class_id, image_shape=image_shape, polygon=polygon
        )
        assert result == expected_result


def test_detections_to_yolo_annotations_raises_for_non_integer_class_id() -> None:
    detections = Detections(
        xyxy=np.array([[100, 100, 200, 200]], dtype=np.float32),
        class_id=np.array([1.9], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="must be an integer"):
        detections_to_yolo_annotations(
            detections=detections, image_shape=(1000, 1000, 3)
        )


def test_load_yolo_annotations_obb_does_not_generate_masks() -> None:
    """OBB annotations must not produce mask arrays (memory regression test)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        images_dir = os.path.join(tmp_dir, "images")
        labels_dir = os.path.join(tmp_dir, "labels")
        os.makedirs(images_dir)
        os.makedirs(labels_dir)

        # Create a small RGB image
        img = Image.new("RGB", (100, 100))
        img.save(os.path.join(images_dir, "test.jpg"))

        # OBB annotation: class_id x1 y1 x2 y2 x3 y3 x4 y4 (9 values per line)
        with open(os.path.join(labels_dir, "test.txt"), "w") as f:
            f.write("0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n")

        # Create a minimal data.yaml
        data_yaml_path = os.path.join(tmp_dir, "data.yaml")
        with open(data_yaml_path, "w") as f:
            f.write("names: ['object']\n")

        _, _, annotations = load_yolo_annotations(
            images_directory_path=images_dir,
            annotations_directory_path=labels_dir,
            data_yaml_path=data_yaml_path,
            is_obb=True,
        )

        assert len(annotations) == 1
        detection = next(iter(annotations.values()))
        assert detection.mask is None, (
            "OBB annotations must not produce mask arrays to avoid excessive memory use"
        )


def test_load_yolo_annotations_obb_force_masks_ignored() -> None:
    """force_masks=True must have no effect when is_obb=True."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        images_dir = os.path.join(tmp_dir, "images")
        labels_dir = os.path.join(tmp_dir, "labels")
        os.makedirs(images_dir)
        os.makedirs(labels_dir)

        img = Image.new("RGB", (100, 100))
        img.save(os.path.join(images_dir, "test.jpg"))

        with open(os.path.join(labels_dir, "test.txt"), "w") as f:
            f.write("0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n")

        data_yaml_path = os.path.join(tmp_dir, "data.yaml")
        with open(data_yaml_path, "w") as f:
            f.write("names: ['object']\n")

        _, _, annotations = load_yolo_annotations(
            images_directory_path=images_dir,
            annotations_directory_path=labels_dir,
            data_yaml_path=data_yaml_path,
            is_obb=True,
            force_masks=True,
        )

        assert len(annotations) == 1
        detection = next(iter(annotations.values()))
        assert detection.mask is None, (
            "force_masks=True must be ignored for OBB annotations"
        )


def test_load_yolo_annotations_segmentation_produces_masks() -> None:
    """Segmentation annotations with is_obb=False must still produce masks."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        images_dir = os.path.join(tmp_dir, "images")
        labels_dir = os.path.join(tmp_dir, "labels")
        os.makedirs(images_dir)
        os.makedirs(labels_dir)

        img = Image.new("RGB", (100, 100))
        img.save(os.path.join(images_dir, "test.jpg"))

        # Polygon annotation: class_id + 3 x,y pairs (7 tokens > 5 triggers mask)
        with open(os.path.join(labels_dir, "test.txt"), "w") as f:
            f.write("0 0.1 0.1 0.9 0.1 0.9 0.9\n")

        data_yaml_path = os.path.join(tmp_dir, "data.yaml")
        with open(data_yaml_path, "w") as f:
            f.write("names: ['object']\n")

        _, _, annotations = load_yolo_annotations(
            images_directory_path=images_dir,
            annotations_directory_path=labels_dir,
            data_yaml_path=data_yaml_path,
            is_obb=False,
        )

        assert len(annotations) == 1
        detection = next(iter(annotations.values()))
        assert detection.mask is not None, (
            "Segmentation annotations with is_obb=False must produce mask arrays"
        )


def test_polygons_to_masks_multiple_polygons_shape() -> None:
    """Regression test for #1746: _polygons_to_masks must return shape (N, H, W).

    The original PR rewrite processed only a single polygon and always returned
    shape (1, H, W), breaking multi-polygon detections.
    """
    from supervision.dataset.formats.yolo import _polygons_to_masks

    resolution_wh = (100, 100)
    # Fractional pixel coords ensure the rounding path inside the function is exercised
    polygon_a = np.array(
        [[10.5, 20.5], [10.5, 50.5], [40.5, 50.5], [40.5, 20.5]], dtype=np.float32
    )
    polygon_b = np.array(
        [[60.3, 30.7], [60.3, 70.3], [90.3, 70.3], [90.3, 30.7]], dtype=np.float32
    )

    masks = _polygons_to_masks(
        polygons=[polygon_a, polygon_b], resolution_wh=resolution_wh
    )

    assert masks.shape == (2, 100, 100), f"Expected (2, 100, 100), got {masks.shape}"
    assert masks.dtype == np.bool_
    assert masks[0].any(), "Polygon A produced an empty mask"
    assert masks[1].any(), "Polygon B produced an empty mask"
    assert not np.any(masks[0] & masks[1]), (
        "Non-overlapping polygons produced overlapping masks"
    )


@pytest.fixture
def yolo_mask_round_trip_sample(
    tmp_path: Path,
) -> tuple[str, str, str, tuple[int, int], str]:
    """Create a minimal YOLO segmentation sample for round-trip mask tests."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    # Odd resolution ensures coord * dim is non-integer (e.g. 0.25 * 101 = 25.25)
    resolution_wh = (101, 97)
    Image.new("RGB", resolution_wh).save(images_dir / "test.jpg")

    original_line = "0 0.25000 0.40000 0.25000 0.60000 0.45000 0.60000 0.45000 0.40000"
    (labels_dir / "test.txt").write_text(original_line + "\n")

    data_yaml_path = tmp_path / "data.yaml"
    data_yaml_path.write_text("names: ['class0']\n")

    return (
        str(images_dir),
        str(labels_dir),
        str(data_yaml_path),
        resolution_wh,
        original_line,
    )


def test_yolo_polygon_mask_precision_no_coord_drift_loads_mask(
    yolo_mask_round_trip_sample: tuple[str, str, str, tuple[int, int], str],
) -> None:
    """YOLO load with force_masks=True should produce a non-empty mask."""
    images_dir, labels_dir, data_yaml_path, _, _ = yolo_mask_round_trip_sample

    _, _, annotations = load_yolo_annotations(
        images_directory_path=images_dir,
        annotations_directory_path=labels_dir,
        data_yaml_path=data_yaml_path,
        force_masks=True,
    )

    assert len(annotations) == 1
    detection = next(iter(annotations.values()))
    assert detection.mask is not None
    assert detection.mask.shape[0] == 1
    assert detection.mask[0].any()


def test_yolo_polygon_mask_precision_no_coord_drift_round_trip_iou(
    yolo_mask_round_trip_sample: tuple[str, str, str, tuple[int, int], str],
) -> None:
    """YOLO load/save round-trip should keep segmentation mask geometry stable."""
    images_dir, labels_dir, data_yaml_path, resolution_wh, original_line = (
        yolo_mask_round_trip_sample
    )

    _, _, annotations = load_yolo_annotations(
        images_directory_path=images_dir,
        annotations_directory_path=labels_dir,
        data_yaml_path=data_yaml_path,
        force_masks=True,
    )
    detection = next(iter(annotations.values()))

    image_arr = np.zeros((resolution_wh[1], resolution_wh[0], 3), dtype=np.uint8)
    saved_lines = detections_to_yolo_annotations(
        detections=detection, image_shape=image_arr.shape
    )

    assert len(saved_lines) == 1
    original_detection = yolo_annotations_to_detections(
        lines=[original_line], resolution_wh=resolution_wh, with_masks=True
    )
    saved_detection = yolo_annotations_to_detections(
        lines=saved_lines, resolution_wh=resolution_wh, with_masks=True
    )

    assert original_detection.mask is not None
    assert saved_detection.mask is not None

    original_mask = original_detection.mask[0]
    saved_mask = saved_detection.mask[0]
    intersection = np.logical_and(original_mask, saved_mask).sum()
    union = np.logical_or(original_mask, saved_mask).sum()
    assert union > 0
    # Keep polygon round-trip drift bounded while avoiding vertex-order assumptions.
    iou = intersection / union
    assert iou > 0.95, (
        f"Mask IoU {iou:.6f} too low after YOLO load/save round-trip — "
        "precision regression in polygon mask conversion"
    )
