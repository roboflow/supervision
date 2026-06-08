from __future__ import annotations

from contextlib import ExitStack as DoesNotRaise
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.dataset.core import DetectionDataset
from supervision.dataset.formats.yolo import (
    _extract_class_names,
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
    ("yaml_text", "expected_names", "exception"),
    [
        (
            "names:\n  '0': background\n  '1': person\n"
            "  '2': car\n  '10': traffic_light\n",
            ["background", "person", "car", "traffic_light"],
            DoesNotRaise(),
        ),  # quoted string numeric keys sort by integer value, not lexicographically
        (
            "names:\n  0: background\n  2: car\n  10: traffic_light\n",
            ["background", "car", "traffic_light"],
            DoesNotRaise(),
        ),  # native int keys (most common YOLO format from Ultralytics/Roboflow)
        (
            "names:\n  cat: 0\n  dog: 1\n",
            ["0", "1"],
            DoesNotRaise(),
        ),  # non-numeric string keys fall back to lexicographic sort
        (
            "names: {}\n",
            [],
            DoesNotRaise(),
        ),  # empty names dict returns empty list
        (
            "names:\n  '--1': ignore\n  '0': person\n",
            None,
            pytest.raises(ValueError, match="mix"),
        ),  # mixed numeric/non-numeric keys raise ValueError
    ],
)
def test_extract_class_names_sorts_numeric_string_keys(
    tmp_path: Path,
    yaml_text: str,
    expected_names: list[str] | None,
    exception: Exception,
) -> None:
    """_extract_class_names returns class names sorted by class index."""
    data_yaml_path = tmp_path / "data.yaml"
    data_yaml_path.write_text(yaml_text, encoding="utf-8")
    with exception:
        assert _extract_class_names(file_path=str(data_yaml_path)) == expected_names


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


@pytest.mark.parametrize(
    ("annotation_line", "load_kwargs", "expect_mask"),
    [
        pytest.param(
            "0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6",
            {"is_obb": True},
            False,
            id="obb-no-mask",
        ),
        pytest.param(
            "0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6",
            {"is_obb": True, "force_masks": True},
            False,
            id="obb-force_masks-ignored",
        ),
        pytest.param(
            "0 0.1 0.1 0.9 0.1 0.9 0.9",
            {"is_obb": False},
            True,
            id="segmentation-produces-mask",
        ),
    ],
)
def test_load_yolo_annotations_mask_behaviour(
    tmp_path: Path,
    annotation_line: str,
    load_kwargs: dict,
    expect_mask: bool,
) -> None:
    """Mask presence depends on annotation format and OBB/segmentation flag."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (100, 100)).save(images_dir / "test.jpg")
    (labels_dir / "test.txt").write_text(annotation_line + "\n")
    (tmp_path / "data.yaml").write_text("names: ['object']\n")

    _, _, annotations = load_yolo_annotations(
        images_directory_path=str(images_dir),
        annotations_directory_path=str(labels_dir),
        data_yaml_path=str(tmp_path / "data.yaml"),
        **load_kwargs,
    )

    detection = next(iter(annotations.values()))
    assert (detection.mask is not None) == expect_mask


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


def test_detections_to_yolo_annotations_obb_emits_nine_tokens() -> None:
    """`is_obb=True` must serialize the 4 corners from `data['xyxyxyxy']`."""
    corners = np.array(
        [[[50.0, 10.0], [90.0, 50.0], [50.0, 90.0], [10.0, 50.0]]], dtype=np.float32
    )
    detections = Detections(
        xyxy=np.array([[10.0, 10.0, 90.0, 90.0]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
        data={ORIENTED_BOX_COORDINATES: corners},
    )

    lines = detections_to_yolo_annotations(
        detections=detections, image_shape=(100, 100, 3), is_obb=True
    )

    assert len(lines) == 1
    tokens = lines[0].split()
    assert len(tokens) == 9, (
        f"OBB export must produce 9 tokens (class + 4 (x,y) pairs), got {tokens}"
    )
    assert tokens[0] == "0"
    np.testing.assert_allclose(
        np.array(tokens[1:], dtype=np.float32),
        np.array([0.5, 0.1, 0.9, 0.5, 0.5, 0.9, 0.1, 0.5], dtype=np.float32),
        atol=1e-5,
    )


def test_detections_to_yolo_annotations_obb_raises_without_corners() -> None:
    """`is_obb=True` without `'xyxyxyxy'` in data must fail loudly, not silently."""
    detections = Detections(
        xyxy=np.array([[10.0, 10.0, 90.0, 90.0]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
    )
    with pytest.raises(ValueError, match=ORIENTED_BOX_COORDINATES):
        detections_to_yolo_annotations(
            detections=detections, image_shape=(100, 100, 3), is_obb=True
        )


def test_detections_to_yolo_annotations_obb_empty_emits_no_lines() -> None:
    """`is_obb=True` on an empty `Detections` must not raise — image had no labels."""
    lines = detections_to_yolo_annotations(
        detections=Detections.empty(), image_shape=(100, 100, 3), is_obb=True
    )
    assert lines == []


def test_detections_to_yolo_annotations_obb_multiple_detections() -> None:
    """OBB export must correctly serialize N>1 detections per image."""
    corners = np.array(
        [
            [[50.0, 10.0], [90.0, 50.0], [50.0, 90.0], [10.0, 50.0]],
            [[20.0, 20.0], [80.0, 20.0], [80.0, 40.0], [20.0, 40.0]],
        ],
        dtype=np.float32,
    )
    detections = Detections(
        xyxy=np.array(
            [[10.0, 10.0, 90.0, 90.0], [20.0, 20.0, 80.0, 40.0]], dtype=np.float32
        ),
        class_id=np.array([0, 1], dtype=int),
        data={ORIENTED_BOX_COORDINATES: corners},
    )

    lines = detections_to_yolo_annotations(
        detections=detections, image_shape=(100, 100, 3), is_obb=True
    )

    assert len(lines) == 2, f"Expected 2 annotation lines, got {len(lines)}"
    for i, line in enumerate(lines):
        tokens = line.split()
        assert len(tokens) == 9, (
            f"Detection {i}: expected 9 tokens, got {len(tokens)}: {tokens}"
        )
    assert lines[0].split()[0] == "0"
    assert lines[1].split()[0] == "1"
    np.testing.assert_allclose(
        np.array(lines[1].split()[1:], dtype=np.float32),
        np.array([0.2, 0.2, 0.8, 0.2, 0.8, 0.4, 0.2, 0.4], dtype=np.float32),
        atol=1e-5,
    )


@pytest.mark.parametrize(
    ("is_obb_save", "expected_tokens"),
    [
        pytest.param(True, 9, id="obb-save-nine-tokens"),
        pytest.param(False, 5, id="default-save-five-tokens"),
    ],
)
def test_dataset_as_yolo_obb_output_token_count(
    tmp_path: Path, is_obb_save: bool, expected_tokens: int
) -> None:
    """Token count in saved YOLO line reflects the is_obb flag at export time."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (100, 100)).save(images_dir / "test.jpg")
    (labels_dir / "test.txt").write_text("0 0.5 0.1 0.9 0.5 0.5 0.9 0.1 0.5\n")
    (tmp_path / "data.yaml").write_text("names: ['object']\n")

    loaded = DetectionDataset.from_yolo(
        images_directory_path=str(images_dir),
        annotations_directory_path=str(labels_dir),
        data_yaml_path=str(tmp_path / "data.yaml"),
        is_obb=True,
    )

    out_labels_dir = tmp_path / "out_labels"
    loaded.as_yolo(
        annotations_directory_path=str(out_labels_dir),
        data_yaml_path=str(tmp_path / "out_data.yaml"),
        is_obb=is_obb_save,
    )

    tokens = (out_labels_dir / "test.txt").read_text().split()
    assert len(tokens) == expected_tokens, (
        f"expected {expected_tokens}-token line, got {len(tokens)}: {tokens}"
    )


def test_dataset_as_yolo_obb_round_trip_corner_accuracy(tmp_path: Path) -> None:
    """OBB round-trip via `from_yolo` -> `as_yolo` must preserve the 4 corners."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    # Non-square image and rotated rhombus exercise both axes.
    Image.new("RGB", (200, 100)).save(images_dir / "test.jpg")
    (labels_dir / "test.txt").write_text("0 0.5 0.1 0.9 0.5 0.5 0.9 0.1 0.5\n")
    (tmp_path / "data.yaml").write_text("names: ['object']\n")

    loaded = DetectionDataset.from_yolo(
        images_directory_path=str(images_dir),
        annotations_directory_path=str(labels_dir),
        data_yaml_path=str(tmp_path / "data.yaml"),
        is_obb=True,
    )

    out_labels_dir = tmp_path / "out_labels"
    loaded.as_yolo(
        annotations_directory_path=str(out_labels_dir),
        data_yaml_path=str(tmp_path / "out_data.yaml"),
        is_obb=True,
    )

    reloaded = DetectionDataset.from_yolo(
        images_directory_path=str(images_dir),
        annotations_directory_path=str(out_labels_dir),
        data_yaml_path=str(tmp_path / "out_data.yaml"),
        is_obb=True,
    )
    original = next(iter(loaded.annotations.values()))
    round_tripped = next(iter(reloaded.annotations.values()))
    np.testing.assert_allclose(
        round_tripped.data[ORIENTED_BOX_COORDINATES],
        original.data[ORIENTED_BOX_COORDINATES],
        atol=1e-3,
    )


def test_dataset_as_yolo_obb_round_trip_with_background_image(
    tmp_path: Path,
) -> None:
    """OBB round-trip with a label-less image must not raise ValueError."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (100, 100)).save(images_dir / "annotated.jpg")
    Image.new("RGB", (100, 100)).save(images_dir / "background.jpg")
    (labels_dir / "annotated.txt").write_text("0 0.5 0.1 0.9 0.5 0.5 0.9 0.1 0.5\n")
    # No labels/background.txt — background image has no annotations.
    (tmp_path / "data.yaml").write_text("names: ['object']\n")

    loaded = DetectionDataset.from_yolo(
        images_directory_path=str(images_dir),
        annotations_directory_path=str(labels_dir),
        data_yaml_path=str(tmp_path / "data.yaml"),
        is_obb=True,
    )

    out_labels_dir = tmp_path / "out_labels"
    loaded.as_yolo(
        annotations_directory_path=str(out_labels_dir),
        data_yaml_path=str(tmp_path / "out_data.yaml"),
        is_obb=True,
    )

    bg_label = out_labels_dir / "background.txt"
    assert bg_label.exists(), "Background image must produce a label file"
    assert bg_label.read_text().strip() == "", (
        "Background image label file must be empty"
    )
