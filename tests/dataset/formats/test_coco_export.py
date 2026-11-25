"""Tests for COCO export and annotation saving."""

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from contextlib import ExitStack as DoesNotRaise
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import numpy.typing as npt
import pytest

import supervision.dataset.utils as dataset_utils
from supervision import DetectionDataset, Detections
from supervision import _cv2 as cv2
from supervision.dataset.formats.coco import (
    classes_to_coco_categories,
    coco_annotations_to_detections,
    coco_annotations_to_masks,
    detections_to_coco_annotations,
    save_coco_annotations,
)
from tests.dataset.formats.helpers import mock_coco_annotation


@pytest.mark.parametrize(
    ("detections", "image_id", "annotation_id", "expected_result", "exception"),
    [
        (
            Detections(
                xyxy=np.array([[0, 0, 100, 100]], dtype=np.float32),
                class_id=np.array([0], dtype=int),
            ),
            0,
            0,
            [
                mock_coco_annotation(
                    category_id=1, bbox=(0, 0, 100, 100), area=100 * 100
                )
            ],
            DoesNotRaise(),
        ),  # no segmentation mask; internal class_id 0 -> COCO category_id 1
        (
            Detections(
                xyxy=np.array([[0, 0, 4, 5]], dtype=np.float32),
                class_id=np.array([0], dtype=int),
                mask=np.array(
                    [
                        [
                            [1, 1, 1, 1, 0],
                            [1, 1, 1, 1, 0],
                            [1, 1, 1, 1, 0],
                            [1, 1, 1, 1, 0],
                            [1, 1, 1, 1, 0],
                        ]
                    ],
                    dtype=bool,
                ),
            ),
            0,
            0,
            [
                mock_coco_annotation(
                    category_id=1,
                    bbox=(0, 0, 4, 5),
                    area=4 * 5,
                    segmentation=[[0, 0, 0, 4, 3, 4, 3, 0]],
                )
            ],
            DoesNotRaise(),
        ),  # segmentation mask in single component,no holes in mask,
        # expects polygon mask
        (
            Detections(
                xyxy=np.array([[0, 0, 5, 5]], dtype=np.float32),
                class_id=np.array([0], dtype=int),
                mask=np.array(
                    [
                        [
                            [1, 1, 1, 0, 0],
                            [1, 1, 1, 0, 0],
                            [1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                        ]
                    ],
                    dtype=bool,
                ),
            ),
            0,
            0,
            [
                mock_coco_annotation(
                    category_id=1,
                    bbox=(0, 0, 5, 5),
                    area=13,
                    segmentation={
                        "size": [5, 5],
                        "counts": [0, 3, 2, 3, 2, 3, 5, 2, 3, 2],
                    },
                    iscrowd=True,
                )
            ],
            DoesNotRaise(),
        ),  # segmentation mask with 2 components, no holes in mask, expects RLE mask
        (
            Detections(
                xyxy=np.array([[0, 0, 5, 5]], dtype=np.float32),
                class_id=np.array([0], dtype=int),
                mask=np.array(
                    [
                        [
                            [0, 1, 1, 1, 1],
                            [0, 1, 1, 1, 1],
                            [1, 1, 0, 0, 1],
                            [1, 1, 0, 0, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ],
                    dtype=bool,
                ),
            ),
            0,
            0,
            [
                mock_coco_annotation(
                    category_id=1,
                    bbox=(0, 0, 5, 5),
                    area=19,
                    segmentation={
                        "size": [5, 5],
                        "counts": [2, 10, 2, 3, 2, 6],
                    },
                    iscrowd=True,
                )
            ],
            DoesNotRaise(),
        ),  # seg mask in single component, with holes in mask, expects RLE mask
    ],
)
def test_detections_to_coco_annotations(
    detections: Detections,
    image_id: int,
    annotation_id: int,
    expected_result: list[dict[str, Any]],
    exception: AbstractContextManager[object],
) -> None:
    with exception:
        result, _ = detections_to_coco_annotations(
            detections=detections,
            image_id=image_id,
            annotation_id=annotation_id,
        )
        assert result == expected_result


def test_detections_to_coco_annotations_handles_empty_approximated_polygons() -> None:
    detections = Detections(
        xyxy=np.array([[0, 0, 4, 4]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
        mask=np.array(
            [
                [
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                ]
            ],
            dtype=bool,
        ),
    )

    with pytest.warns(Warning, match="mask approximation returned no polygons"):
        annotations, _ = detections_to_coco_annotations(
            detections=detections,
            image_id=0,
            annotation_id=0,
            max_image_area_percentage=0.01,
        )

    assert len(annotations) == 1
    assert annotations[0]["segmentation"] == []
    assert annotations[0]["iscrowd"] == 0


_DISJOINT_2X2_MASK = np.array(
    [
        [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
        ]
    ],
    dtype=bool,
)


_SINGLE_COMPONENT_MASK = np.array(
    [
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    ],
    dtype=bool,
)


def _make_iscrowd0_detections(mask: np.ndarray) -> Detections:
    """Build a single-detection Detections with iscrowd=0 from a (1, H, W) mask."""
    _, h, w = mask.shape
    return Detections(
        xyxy=np.array([[0, 0, w, h]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
        mask=mask,
        data={"iscrowd": np.array([0])},
    )


@pytest.mark.parametrize(
    ("mask", "expected_segment_count"),
    [
        pytest.param(_DISJOINT_2X2_MASK, 2, id="disjoint-two-parts"),
        pytest.param(_SINGLE_COMPONENT_MASK, 1, id="single-component"),
    ],
)
def test_detections_to_coco_annotations_segmentation_count(
    mask: np.ndarray, expected_segment_count: int
) -> None:
    """Non-crowd mask export produces one polygon list entry per connected component."""
    annotations, _ = detections_to_coco_annotations(
        detections=_make_iscrowd0_detections(mask), image_id=0, annotation_id=0
    )

    segmentation = annotations[0]["segmentation"]
    assert annotations[0]["iscrowd"] == 0
    assert isinstance(segmentation, list)
    assert len(segmentation) == expected_segment_count
    assert all(len(part) >= 6 for part in segmentation)
    assert all(np.isfinite(c) for part in segmentation for c in part)


def test_detections_to_coco_annotations_round_trip_disjoint_mask() -> None:
    """Two-part disjoint mask round-trips through COCO export and import unchanged."""
    W, H = 5, 5
    annotations, _ = detections_to_coco_annotations(
        detections=_make_iscrowd0_detections(_DISJOINT_2X2_MASK),
        image_id=0,
        annotation_id=0,
    )
    reloaded = coco_annotations_to_masks([annotations[0]], resolution_wh=(W, H))

    assert np.array_equal(reloaded[0], _DISJOINT_2X2_MASK[0])


def _mask_with_hole() -> npt.NDArray[np.bool_]:
    """Return a 40x40 mask with one 10x10 hole."""
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:35, 5:35] = True
    mask[15:25, 15:25] = False
    return mask


def _mask_with_island_in_hole() -> npt.NDArray[np.bool_]:
    """Return a 40x40 mask whose hole contains a separate island."""
    mask = _mask_with_hole()
    mask[18:22, 18:22] = True
    return mask


def _mask_with_nested_holes() -> npt.NDArray[np.bool_]:
    """Return a 40x40 mask with a hole enclosed by a C-shaped hole."""
    mask = np.zeros((40, 40), dtype=bool)
    mask[2:38, 2:38] = True
    mask[8:32, 8:32] = False
    mask[12:28, 12:28] = True
    mask[18:22, 28:32] = True
    mask[18:22, 18:22] = False
    return mask


class TestDetectionsToCocoAnnotationsMaskHoles:
    """Non-crowd masks keep their holes through a COCO polygon export."""

    @pytest.mark.parametrize(
        "make_mask",
        [
            pytest.param(_mask_with_hole, id="hole"),
            pytest.param(_mask_with_island_in_hole, id="island-in-hole"),
            pytest.param(_mask_with_nested_holes, id="nested-holes"),
        ],
    )
    def test_round_trip_restores_mask(
        self, make_mask: Callable[[], npt.NDArray[np.bool_]]
    ) -> None:
        """Export and import give back the same mask, holes included."""
        mask = make_mask()
        annotations, _ = detections_to_coco_annotations(
            detections=_make_iscrowd0_detections(mask[np.newaxis]),
            image_id=0,
            annotation_id=0,
            approximation_percentage=0.0,
        )

        reloaded = coco_annotations_to_masks([annotations[0]], resolution_wh=(40, 40))

        np.testing.assert_array_equal(reloaded[0], mask)

    def test_unbridgeable_hole_warns_and_keeps_exporting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hole with no seam stays a separate part and does not abort the export."""

        def find_no_seam(*args: Any, **kwargs: Any) -> NoReturn:
            """Stand in for a seam search that finds nothing."""
            raise dataset_utils._SeamNotFoundError

        monkeypatch.setattr(dataset_utils, "_find_seam", find_no_seam)
        mask = _mask_with_hole()

        with pytest.warns(UserWarning, match="Could not join the holes"):
            annotations, _ = detections_to_coco_annotations(
                detections=_make_iscrowd0_detections(mask[np.newaxis]),
                image_id=0,
                annotation_id=0,
                approximation_percentage=0.0,
            )

        assert len(annotations[0]["segmentation"]) == 2

    def test_invalid_approximation_percentage_raises_without_warning(
        self, recwarn: pytest.WarningsRecorder
    ) -> None:
        """A bad argument raises as such, not as a hole that could not be joined."""
        mask = _mask_with_hole()

        with pytest.raises(ValueError, match="Percentage must be"):
            detections_to_coco_annotations(
                detections=_make_iscrowd0_detections(mask[np.newaxis]),
                image_id=0,
                annotation_id=0,
                approximation_percentage=1.5,
            )

        assert not recwarn.list

    @pytest.mark.filterwarnings("ignore:__array__ implementation:DeprecationWarning")
    def test_pycocotools_keeps_hole_open(self) -> None:
        """A reference COCO reader sees the hole and keeps the foreground around it."""
        coco_mask = pytest.importorskip("pycocotools.mask")
        mask = _mask_with_hole()
        annotations, _ = detections_to_coco_annotations(
            detections=_make_iscrowd0_detections(mask[np.newaxis]),
            image_id=0,
            annotation_id=0,
            approximation_percentage=0.0,
        )

        rles = coco_mask.frPyObjects(annotations[0]["segmentation"], 40, 40)
        decoded = coco_mask.decode(coco_mask.merge(rles)).astype(bool)

        # Contours lie on foreground pixels and pycocotools reads them as continuous
        # coordinates, so the traced hole misses one corner pixel. The seam paints none.
        assert int(decoded[15:25, 15:25].sum()) == 1
        assert decoded[mask].mean() > 0.85


def test_detections_to_coco_annotations_preserves_area_from_data() -> None:
    """Area stored in detections.data should be used instead of bbox area."""
    detections = Detections(
        xyxy=np.array([[10.0, 20.0, 110.0, 120.0]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
        data={"iscrowd": np.array([0], dtype=int), "area": np.array([5000.0])},
    )

    annotations, _ = detections_to_coco_annotations(
        detections=detections,
        image_id=1,
        annotation_id=1,
    )

    assert len(annotations) == 1
    assert annotations[0]["area"] == 5000.0
    assert annotations[0]["iscrowd"] == 0
    assert type(annotations[0]["iscrowd"]) is int


def test_detections_to_coco_annotations_preserves_iscrowd_from_data_when_no_mask() -> (
    None
):
    """Iscrowd stored in detections.data should be used when no mask is present."""
    detections = Detections(
        xyxy=np.array([[0.0, 0.0, 100.0, 100.0]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
        data={"iscrowd": np.array([1], dtype=int), "area": np.array([1234.5])},
    )

    annotations, _ = detections_to_coco_annotations(
        detections=detections,
        image_id=1,
        annotation_id=1,
    )

    assert len(annotations) == 1
    assert annotations[0]["iscrowd"] == 1
    assert type(annotations[0]["iscrowd"]) is int
    assert annotations[0]["area"] == 1234.5


def test_detections_to_coco_annotations_iscrowd_is_int_when_mask_provided() -> None:
    """Iscrowd should be stored as int (0 or 1), not as Python bool."""
    mask = np.zeros((1, 5, 5), dtype=bool)
    mask[0, 0:3, 0:3] = True  # simple single-component rectangle

    detections = Detections(
        xyxy=np.array([[0.0, 0.0, 3.0, 3.0]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
        mask=mask,
    )

    annotations, _ = detections_to_coco_annotations(
        detections=detections,
        image_id=1,
        annotation_id=1,
    )

    assert len(annotations) == 1
    assert annotations[0]["iscrowd"] == 0
    assert type(annotations[0]["iscrowd"]) is int


def test_detections_to_coco_annotations_data_area_overrides_bbox_with_mask() -> None:
    """Data["area"] should override computed bbox area even when a mask is present."""
    mask = np.zeros((1, 10, 10), dtype=bool)
    mask[0, 0:4, 0:4] = True  # 16-pixel polygon area

    detections = Detections(
        xyxy=np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
        mask=mask,
        data={"area": np.array([999.0])},
    )

    annotations, _ = detections_to_coco_annotations(
        detections=detections,
        image_id=1,
        annotation_id=1,
    )

    assert len(annotations) == 1
    assert annotations[0]["area"] == 999.0


def test_detections_to_coco_annotations_mask_area_when_no_data() -> None:
    """Masked detections without stored area export pixel area, not bbox area."""
    mask = np.zeros((1, 10, 10), dtype=bool)
    mask[0, 0:4, 0:4] = True

    detections = Detections(
        xyxy=np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
        mask=mask,
    )

    annotations, _ = detections_to_coco_annotations(
        detections=detections,
        image_id=1,
        annotation_id=1,
    )

    assert len(annotations) == 1
    assert annotations[0]["area"] == 16.0


def test_missing_coco_area_with_mask_exports_mask_pixel_area() -> None:
    """Segmented COCO annotations without area export decoded mask area."""
    detections = coco_annotations_to_detections(
        image_annotations=[
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "segmentation": [[0, 0, 5, 0, 5, 5, 0, 5]],
            }
        ],
        resolution_wh=(10, 10),
        with_masks=True,
        use_iscrowd=True,
    )
    assert detections.mask is not None
    expected_area = float(np.count_nonzero(detections.mask[0]))

    annotations, _ = detections_to_coco_annotations(
        detections=detections,
        image_id=1,
        annotation_id=1,
    )

    assert annotations[0]["area"] == expected_area
    assert annotations[0]["area"] != 100.0


def test_detections_to_coco_annotations_fallback_area_when_no_data() -> None:
    """Box-only detections with no area in data fall back to bbox area."""
    detections = Detections(
        xyxy=np.array([[10.0, 20.0, 110.0, 120.0]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
    )

    annotations, _ = detections_to_coco_annotations(
        detections=detections,
        image_id=1,
        annotation_id=1,
    )

    assert len(annotations) == 1
    assert annotations[0]["area"] == 100.0 * 100.0
    assert annotations[0]["iscrowd"] == 0


# --- category_id 1-indexing (regression for #1181) ---


@pytest.mark.parametrize(
    ("classes", "expected_ids"),
    [
        ([], []),  # empty classes
        (["object"], [1]),  # single class starts at 1
        (["cat", "dog", "bird"], [1, 2, 3]),  # ids are sequential and 1-indexed
    ],
)
def test_classes_to_coco_categories_ids_start_at_one(
    classes: list[str], expected_ids: list[int]
) -> None:
    """COCO categories[].id must be 1-indexed (COCO spec / CVAT requirement)."""
    categories = classes_to_coco_categories(classes=classes)

    assert [category["id"] for category in categories] == expected_ids


def test_detections_to_coco_annotations_category_id_is_one_indexed() -> None:
    """Internal class_id k must serialize to COCO category_id k + 1."""
    detections = Detections(
        xyxy=np.array([[0, 0, 10, 10], [5, 5, 15, 15], [1, 1, 4, 4]], dtype=np.float32),
        class_id=np.array([0, 1, 2], dtype=int),
    )

    annotations, _ = detections_to_coco_annotations(
        detections=detections,
        image_id=1,
        annotation_id=1,
    )

    assert [annotation["category_id"] for annotation in annotations] == [1, 2, 3]


def test_coco_round_trip_preserves_class_ids_and_writes_one_indexed_categories(
    tmp_path: Path,
) -> None:
    """as_coco -> from_coco is lossless for internal class_ids while the on-disk COCO
    category ids are 1-indexed (regression for #1181)."""
    classes = ["cat", "dog"]
    image_paths: list[str] = []
    annotations: dict[str, Detections] = {}
    expected_class_ids = {}
    for index, class_id in enumerate([0, 1]):
        path = str(tmp_path / f"image_{index}.jpg")
        assert cv2.imwrite(path, np.zeros((10, 10, 3), dtype=np.uint8))
        image_paths.append(path)
        detections = Detections(
            xyxy=np.array([[0, 0, 5, 5]], dtype=np.float32),
            class_id=np.array([class_id], dtype=int),
        )
        annotations[path] = detections
        expected_class_ids[Path(path).name] = class_id
    dataset = DetectionDataset(
        classes=classes, images=image_paths, annotations=annotations
    )

    annotation_path = tmp_path / "annotations.json"
    dataset.as_coco(annotations_path=str(annotation_path))

    # On-disk COCO ids are 1-indexed.
    with open(annotation_path) as f:
        payload = json.load(f)
    assert sorted(category["id"] for category in payload["categories"]) == [1, 2]
    assert sorted(ann["category_id"] for ann in payload["annotations"]) == [1, 2]

    # Reading back preserves internal 0-indexed class_ids losslessly.
    loaded = DetectionDataset.from_coco(
        images_directory_path=str(tmp_path),
        annotations_path=str(annotation_path),
    )
    assert loaded.classes == classes
    for image_path, _, detections in loaded:
        name = Path(image_path).name
        assert detections.class_id is not None
        assert detections.class_id.tolist() == [expected_class_ids[name]]


# --- save_coco_annotations: cross-split id chaining (regression for #768) ---


def _tiny_detection_dataset(
    tmp_path: Path, prefix: str, num_images: int, dets_per_image: int
) -> DetectionDataset:
    """Build a DetectionDataset of ``num_images`` 10x10 RGB images on disk, each holding
    ``dets_per_image`` 1x1 detections of class 0.

    Image content is irrelevant; only the per-image Detections drive the COCO write
    path.
    """
    classes = ["object"]
    image_paths: list[str] = []
    annotations: dict[str, Detections] = {}
    for i in range(num_images):
        path = str(tmp_path / f"{prefix}_{i}.jpg")
        assert cv2.imwrite(path, np.zeros((10, 10, 3), dtype=np.uint8))
        image_paths.append(path)
        xyxy = np.array(
            [[float(j), 0.0, float(j) + 1.0, 1.0] for j in range(dets_per_image)],
            dtype=float,
        ).reshape(-1, 4)
        annotations[path] = Detections(
            xyxy=xyxy,
            class_id=np.zeros(dets_per_image, dtype=int),
            confidence=np.ones(dets_per_image, dtype=float),
        )
    return DetectionDataset(
        classes=classes, images=image_paths, annotations=annotations
    )


def _read_ids(annotation_path: Path) -> tuple[list[int], list[int]]:
    """Read the image and annotation ids written to a COCO JSON file."""
    with open(annotation_path) as f:
        payload = json.load(f)
    image_ids = [img["id"] for img in payload["images"]]
    annotation_ids = [ann["id"] for ann in payload["annotations"]]
    return image_ids, annotation_ids


class TestSaveCocoAnnotationsCollisionGuard:
    """COCO export must reject same-basename images before writing."""

    def test_raises_on_duplicate_image_basenames(self, tmp_path: Path) -> None:
        """Duplicate image basenames are rejected instead of being collapsed."""
        image_paths = []
        annotations: dict[str, Detections] = {}
        for parent in ("dir_a", "dir_b"):
            image_path = tmp_path / parent / "img.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(image_path), np.zeros((10, 10, 3), dtype=np.uint8))
            image_path_str = str(image_path)
            image_paths.append(image_path_str)
            annotations[image_path_str] = Detections.empty()

        dataset = DetectionDataset(
            classes=["object"], images=image_paths, annotations=annotations
        )

        with pytest.raises(ValueError, match="COCO image file"):
            save_coco_annotations(
                dataset=dataset,
                annotation_path=str(tmp_path / "annotations.json"),
            )


def test_save_coco_annotations_defaults_start_at_one(tmp_path: Path) -> None:
    dataset = _tiny_detection_dataset(tmp_path, "img", num_images=2, dets_per_image=3)
    annotation_path = tmp_path / "annotations.json"

    next_image_id, next_annotation_id = save_coco_annotations(
        dataset=dataset, annotation_path=str(annotation_path)
    )

    image_ids, annotation_ids = _read_ids(annotation_path)
    assert image_ids == [1, 2]
    assert annotation_ids == [1, 2, 3, 4, 5, 6]
    # Returned ids are one greater than the highest written, ready to chain.
    assert next_image_id == 3
    assert next_annotation_id == 7


def test_save_coco_annotations_respects_starting_ids(tmp_path: Path) -> None:
    dataset = _tiny_detection_dataset(tmp_path, "img", num_images=2, dets_per_image=2)
    annotation_path = tmp_path / "annotations.json"

    next_image_id, next_annotation_id = save_coco_annotations(
        dataset=dataset,
        annotation_path=str(annotation_path),
        starting_image_id=100,
        starting_annotation_id=500,
    )

    image_ids, annotation_ids = _read_ids(annotation_path)
    assert image_ids == [100, 101]
    assert annotation_ids == [500, 501, 502, 503]
    assert next_image_id == 102
    assert next_annotation_id == 504


def test_as_coco_chains_ids_across_splits_without_collision(tmp_path: Path) -> None:
    """Regression for #768: exporting train/valid/test splits with the returned ids fed
    forward yields globally unique image and annotation ids."""
    train = _tiny_detection_dataset(tmp_path, "train", num_images=3, dets_per_image=2)
    valid = _tiny_detection_dataset(tmp_path, "valid", num_images=2, dets_per_image=4)
    test = _tiny_detection_dataset(tmp_path, "test", num_images=1, dets_per_image=5)

    train_path = tmp_path / "train.json"
    valid_path = tmp_path / "valid.json"
    test_path = tmp_path / "test.json"

    next_image_id, next_annotation_id = train.as_coco(annotations_path=str(train_path))
    next_image_id, next_annotation_id = valid.as_coco(
        annotations_path=str(valid_path),
        starting_image_id=next_image_id,
        starting_annotation_id=next_annotation_id,
    )
    test.as_coco(
        annotations_path=str(test_path),
        starting_image_id=next_image_id,
        starting_annotation_id=next_annotation_id,
    )

    all_image_ids: list[int] = []
    all_annotation_ids: list[int] = []
    for path in (train_path, valid_path, test_path):
        image_ids, annotation_ids = _read_ids(path)
        all_image_ids.extend(image_ids)
        all_annotation_ids.extend(annotation_ids)

    assert len(all_image_ids) == len(set(all_image_ids)), (
        "image ids collide across splits"
    )
    assert len(all_annotation_ids) == len(set(all_annotation_ids)), (
        "annotation ids collide across splits"
    )
    # Concrete chained values.
    assert all_image_ids == [1, 2, 3, 4, 5, 6]
    assert all_annotation_ids == list(range(1, 6 + 8 + 5 + 1))


def test_save_coco_annotations_empty_dataset_returns_starting_ids(
    tmp_path: Path,
) -> None:
    """An empty dataset writes a valid (but empty) COCO file and returns the starting
    ids unchanged so chaining still composes around it."""
    dataset = DetectionDataset(classes=["object"], images=[], annotations={})
    annotation_path = tmp_path / "annotations.json"

    next_image_id, next_annotation_id = save_coco_annotations(
        dataset=dataset,
        annotation_path=str(annotation_path),
        starting_image_id=7,
        starting_annotation_id=42,
    )

    image_ids, annotation_ids = _read_ids(annotation_path)
    assert image_ids == []
    assert annotation_ids == []
    assert next_image_id == 7
    assert next_annotation_id == 42


def test_as_coco_without_annotations_path_returns_starting_ids(tmp_path: Path) -> None:
    """When only writing images, the starting ids round-trip unchanged so chaining still
    works in the images-only branch."""
    dataset = _tiny_detection_dataset(tmp_path, "img", num_images=2, dets_per_image=1)
    next_image_id, next_annotation_id = dataset.as_coco(
        images_directory_path=str(tmp_path / "imgs"),
        starting_image_id=42,
        starting_annotation_id=99,
    )
    assert next_image_id == 42
    assert next_annotation_id == 99


def test_save_coco_annotations_annotation_image_id_references_correct_image(
    tmp_path: Path,
) -> None:
    """Every annotation's image_id must reference an image id present in the same file,
    even when a non-default starting_image_id is used."""
    dataset = _tiny_detection_dataset(tmp_path, "img", num_images=3, dets_per_image=2)
    annotation_path = tmp_path / "annotations.json"

    save_coco_annotations(
        dataset=dataset,
        annotation_path=str(annotation_path),
        starting_image_id=100,
        starting_annotation_id=500,
    )

    with open(annotation_path) as f:
        coco = json.load(f)
    image_id_set = {img["id"] for img in coco["images"]}
    annotation_image_ids = {ann["image_id"] for ann in coco["annotations"]}
    assert annotation_image_ids <= image_id_set, (
        "annotation image_id values reference unknown image ids"
    )


def test_save_coco_annotations_zero_annotation_images(tmp_path: Path) -> None:
    """Dataset with images but zero detections per image: image ids are assigned
    sequentially but annotation list stays empty."""
    dataset = _tiny_detection_dataset(tmp_path, "img", num_images=2, dets_per_image=0)
    annotation_path = tmp_path / "annotations.json"

    next_image_id, next_annotation_id = save_coco_annotations(
        dataset=dataset, annotation_path=str(annotation_path)
    )

    image_ids, annotation_ids = _read_ids(annotation_path)
    assert image_ids == [1, 2]
    assert annotation_ids == []
    assert next_image_id == 3
    assert next_annotation_id == 1


class TestSaveCocoAnnotationsHeaderSizeReads:
    """Annotation export must read image sizes without decoding pixels."""

    def test_labels_only_export_does_not_decode_images(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A labels-only export never loads image pixel data."""

        def _fail_on_decode(self: DetectionDataset, image_path: str) -> None:
            """Fail the test if the export tries to load an image."""
            raise AssertionError(f"export decoded image pixels: {image_path}")

        dataset = _tiny_detection_dataset(
            tmp_path, "img", num_images=2, dets_per_image=1
        )
        monkeypatch.setattr(DetectionDataset, "_get_image", _fail_on_decode)
        annotation_path = tmp_path / "annotations.json"

        save_coco_annotations(dataset=dataset, annotation_path=str(annotation_path))

        image_ids, annotation_ids = _read_ids(annotation_path)
        assert image_ids == [1, 2]
        assert annotation_ids == [1, 2]

    def test_header_sizes_match_image_dimensions(self, tmp_path: Path) -> None:
        """Sizes read from headers match the real (non-square) image shape."""
        image_path = str(tmp_path / "img.jpg")
        assert cv2.imwrite(image_path, np.zeros((8, 12, 3), dtype=np.uint8))
        dataset = DetectionDataset(
            classes=["object"],
            images=[image_path],
            annotations={image_path: Detections.empty()},
        )
        annotation_path = tmp_path / "annotations.json"

        save_coco_annotations(dataset=dataset, annotation_path=str(annotation_path))

        with open(annotation_path) as f:
            coco = json.load(f)
        assert coco["images"][0]["height"] == 8
        assert coco["images"][0]["width"] == 12

    def test_header_sizes_follow_exif_orientation(self, tmp_path: Path) -> None:
        """Sizes follow the EXIF rotation that loading the image applies."""
        from PIL import Image

        image_path = str(tmp_path / "photo.jpg")
        exif = Image.Exif()
        exif[0x0112] = 6
        Image.new("RGB", (12, 8)).save(image_path, exif=exif.tobytes())
        dataset = DetectionDataset(
            classes=["object"],
            images=[image_path],
            annotations={image_path: Detections.empty()},
        )
        annotation_path = tmp_path / "annotations.json"

        save_coco_annotations(dataset=dataset, annotation_path=str(annotation_path))

        with open(annotation_path) as f:
            coco = json.load(f)
        assert coco["images"][0]["height"] == 12
        assert coco["images"][0]["width"] == 8

    def test_in_memory_images_use_array_shape(self, tmp_path: Path) -> None:
        """Datasets built from in-memory arrays take sizes from the arrays."""
        from supervision.utils.internal import SupervisionWarnings

        image_key = "in_memory.jpg"
        with pytest.warns(SupervisionWarnings):
            dataset = DetectionDataset(
                classes=["object"],
                images={image_key: np.zeros((6, 9, 3), dtype=np.uint8)},
                annotations={image_key: Detections.empty()},
            )
        annotation_path = tmp_path / "annotations.json"

        save_coco_annotations(dataset=dataset, annotation_path=str(annotation_path))

        with open(annotation_path) as f:
            coco = json.load(f)
        assert coco["images"][0]["height"] == 6
        assert coco["images"][0]["width"] == 9


# --- save_coco_annotations ValueError guards ---


@pytest.mark.parametrize(
    ("starting_image_id", "starting_annotation_id"),
    [
        (0, 1),
        (1, 0),
        (0, 0),
    ],
)
def test_save_coco_annotations_rejects_zero_starting_ids(
    tmp_path: Path, starting_image_id: int, starting_annotation_id: int
) -> None:
    """starting_image_id and starting_annotation_id below 1 must raise ValueError."""
    dataset = DetectionDataset(classes=["object"], images=[], annotations={})
    annotation_path = tmp_path / "annotations.json"

    with pytest.raises(ValueError, match="must be >= 1"):
        save_coco_annotations(
            dataset=dataset,
            annotation_path=str(annotation_path),
            starting_image_id=starting_image_id,
            starting_annotation_id=starting_annotation_id,
        )


# --- detections_to_coco_annotations: class_id=None guard ---


def test_detections_to_coco_annotations_raises_when_class_id_is_none() -> None:
    """Detections with no class_id must raise ValueError before +1 arithmetic."""
    detections = Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
        class_id=None,
    )

    with pytest.raises(ValueError, match="class_id"):
        detections_to_coco_annotations(
            detections=detections,
            image_id=1,
            annotation_id=1,
        )


# --- Round-trip: multi-class-per-image case ---


def test_coco_round_trip_multi_class_single_image(tmp_path: Path) -> None:
    """Single image with two detections of different classes round-trips losslessly."""
    img_path = str(tmp_path / "img.jpg")
    assert cv2.imwrite(img_path, np.zeros((10, 10, 3), dtype=np.uint8))

    dataset = DetectionDataset(
        classes=["cat", "dog"],
        images=[img_path],
        annotations={
            img_path: Detections(
                xyxy=np.array([[0, 0, 5, 5], [1, 1, 4, 4]], dtype=np.float32),
                class_id=np.array([0, 1], dtype=int),
            )
        },
    )

    annotation_path = tmp_path / "annotations.json"
    dataset.as_coco(annotations_path=str(annotation_path))

    with open(annotation_path) as f:
        payload = json.load(f)
    assert sorted(ann["category_id"] for ann in payload["annotations"]) == [1, 2]

    loaded = DetectionDataset.from_coco(
        images_directory_path=str(tmp_path),
        annotations_path=str(annotation_path),
    )
    dets = loaded.annotations[img_path]
    assert dets.class_id is not None
    assert sorted(dets.class_id.tolist()) == [0, 1]


# --- Regression: segmentation round-trip (#2285) ---


def _coco_annotation_with_segmentation(
    segmentation: list[list[float]],
    bbox: tuple[float, float, float, float] = (0, 0, 5, 5),
    area: float = 25,
) -> dict[str, Any]:
    """Build a single COCO annotation carrying the given polygon segmentation."""
    return mock_coco_annotation(
        annotation_id=1,
        image_id=1,
        category_id=1,
        bbox=bbox,
        area=area,
        segmentation=segmentation,
    )


def _single_image_coco_data(annotation: dict[str, Any]) -> dict[str, object]:
    """Wrap one COCO annotation in a minimal single-image COCO payload."""
    return {
        "info": {},
        "licenses": [],
        "categories": [{"id": 1, "name": "cat", "supercategory": ""}],
        "images": [{"id": 1, "file_name": "img.jpg", "width": 10, "height": 10}],
        "annotations": [annotation],
    }


class TestCocoExportMaskArea:
    @pytest.mark.parametrize(
        ("mask_side", "stored_area", "iscrowd", "expected_area"),
        [
            pytest.param(0, None, 0, 30.0, id="empty-placeholder"),
            pytest.param(0, np.nan, 0, 30.0, id="nonfinite-stored-area"),
            pytest.param(0, 0.0, 0, 0.0, id="explicit-zero"),
            pytest.param(0, 24.0, 0, 24.0, id="explicit-positive-area"),
            pytest.param(0, None, 1, 0.0, id="empty-crowd-rle"),
            pytest.param(4, None, 0, 16.0, id="filtered-nonempty-mask"),
        ],
    )
    def test_uses_area_for_the_available_geometry(
        self,
        mask_side: int,
        stored_area: float | None,
        iscrowd: int,
        expected_area: float,
    ) -> None:
        """Preserve explicit and mask areas while exporting bbox placeholders."""
        mask = np.zeros((1, 10, 10), dtype=bool)
        mask[0, :mask_side, :mask_side] = True
        data: dict[str, npt.NDArray[np.generic] | list[Any]] = {
            "iscrowd": np.array([iscrowd], dtype=int)
        }
        if stored_area is not None:
            data["area"] = np.array([stored_area], dtype=float)
        detections = Detections(
            xyxy=np.array([[1, 1, 7, 6]], dtype=np.float32),
            class_id=np.array([0], dtype=int),
            mask=mask,
            data=data,
        )

        context = (
            DoesNotRaise()
            if iscrowd
            else pytest.warns(
                UserWarning, match="mask approximation returned no polygons"
            )
        )
        with context:
            annotations, _ = detections_to_coco_annotations(
                detections=detections,
                image_id=1,
                annotation_id=1,
                max_image_area_percentage=0.05,
            )

        (annotation,) = annotations
        assert annotation["bbox"] == [1, 1, 6, 5]
        assert annotation["area"] == expected_area
        assert annotation["segmentation"] == (
            {"counts": [100], "size": [10, 10]} if iscrowd else []
        )

    @pytest.mark.parametrize("mixed", [False, True])
    def test_box_area_survives_dataset_round_trip(
        self, tmp_path: Path, mixed: bool
    ) -> None:
        """Retain box area for a round trip using all-zero mask placeholders."""
        image_path = tmp_path / "img.jpg"
        assert cv2.imwrite(str(image_path), np.zeros((10, 10, 3), dtype=np.uint8))
        bbox = [1, 1, 6, 5]
        box_annotation = {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": bbox,
            "segmentation": [],
        }
        annotations = [box_annotation]
        if mixed:
            annotations.append(
                {
                    **box_annotation,
                    "id": 2,
                    "bbox": [6, 6, 3, 3],
                    "segmentation": [[6, 6, 8, 6, 8, 8, 6, 8]],
                }
            )
        source = tmp_path / "source.json"
        source.write_text(
            json.dumps(
                {**_single_image_coco_data(box_annotation), "annotations": annotations}
            )
        )
        dataset = DetectionDataset.from_coco(
            images_directory_path=str(tmp_path),
            annotations_path=str(source),
            force_masks=not mixed,
            use_iscrowd=False,
            mask_rasterizer="supervision",
        )
        detections = dataset.annotations[str(image_path)]
        assert detections.mask is not None
        assert not detections.mask[0].any()
        assert "area" not in detections.data
        target = tmp_path / "target.json"

        with pytest.warns(UserWarning, match="mask approximation returned no polygons"):
            dataset.as_coco(annotations_path=str(target))

        exported = json.loads(target.read_text())["annotations"]
        assert len(exported) == len(annotations)
        assert exported[0]["bbox"] == bbox
        assert exported[0]["segmentation"] == []
        assert exported[0]["area"] == 30.0
        if mixed:
            assert exported[1]["segmentation"]
            assert exported[1]["area"] == 9.0


def test_detections_to_coco_annotations_exports_all_polygons() -> None:
    """All polygons from a multi-component mask must be exported, not just the first."""
    # Build a mask with two separate rectangles (disjoint components)
    mask = np.zeros((20, 20), dtype=bool)
    mask[1:4, 1:4] = True  # top-left component
    mask[14:18, 14:18] = True  # bottom-right component

    detections = Detections(
        xyxy=np.array([[1, 1, 4, 4]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
        mask=np.array([mask]),
        data={"iscrowd": np.array([0], dtype=int)},
    )
    annotations, _ = detections_to_coco_annotations(
        detections=detections, image_id=1, annotation_id=1
    )
    assert len(annotations) == 1
    seg = annotations[0]["segmentation"]
    # Both components must appear as separate polygon entries (list of lists)
    assert isinstance(seg, list), "segmentation must be a list of polygons"
    assert len(seg) >= 2


@pytest.mark.parametrize(
    ("segmentation", "bbox", "area", "expected_min_polygon_count"),
    [
        pytest.param(
            [[0, 0, 4, 0, 4, 4, 0, 4]],
            (0, 0, 5, 5),
            25,
            1,
            id="single-polygon",
        ),
        pytest.param(
            [[0, 0, 4, 0, 4, 4, 0, 4], [6, 6, 9, 6, 9, 9, 6, 9]],
            (0, 0, 9, 9),
            32,
            2,
            id="multi-polygon",
        ),
    ],
)
def test_coco_polygon_segmentation_survives_roundtrip(
    tmp_path: Path,
    segmentation: list[list[float]],
    bbox: tuple[float, float, float, float],
    area: float,
    expected_min_polygon_count: int,
) -> None:
    """COCO polygon segmentation survives the load/export sequence.

    1. Write source COCO JSON with polygon segmentation.
    2. Load it through DetectionDataset.from_coco().
    3. Export it back to COCO JSON with as_coco().
    4. Assert the exported segmentation keeps the expected polygon component count.
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "img.jpg"
    assert cv2.imwrite(str(img_path), np.zeros((10, 10, 3), dtype=np.uint8))

    # 1. Write source COCO JSON with polygon segmentation.
    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(
        json.dumps(
            _single_image_coco_data(
                _coco_annotation_with_segmentation(
                    segmentation=segmentation, bbox=bbox, area=area
                )
            )
        ),
        encoding="utf-8",
    )

    # 2. Load it through the internal DetectionDataset representation.
    ds = DetectionDataset.from_coco(
        images_directory_path=str(images_dir),
        annotations_path=str(ann_path),
        mask_rasterizer="supervision",
    )

    # 3. Export it back to COCO JSON.
    out_ann_path = tmp_path / "out_annotations.json"
    ds.as_coco(annotations_path=str(out_ann_path))

    with open(out_ann_path) as f:
        out = json.load(f)

    # 4. Assert polygon component count survives the load/export sequence.
    assert len(out["annotations"]) == 1
    seg = out["annotations"][0]["segmentation"]
    assert isinstance(seg, list)
    assert len(seg) >= expected_min_polygon_count


def test_coco_raw_segmentation_preserved_when_masks_not_decoded() -> None:
    """When masks are NOT decoded (with_masks=False), raw polygon data stored in
    data['segmentation'] is used as a lossless fallback so as_coco() still emits non-
    empty segmentation."""
    image_annotations = [
        _coco_annotation_with_segmentation(segmentation=[[0, 0, 4, 0, 4, 4, 0, 4]])
    ]

    # Load WITHOUT mask decoding — mask must be None
    detections = coco_annotations_to_detections(
        image_annotations=image_annotations,
        resolution_wh=(10, 10),
        with_masks=False,
    )
    assert detections.mask is None
    # Raw segmentation must be stored in data for fallback
    assert "coco_raw_segmentation" in detections.data

    # Export must still produce non-empty segmentation via fallback
    annotations, _ = detections_to_coco_annotations(
        detections=detections, image_id=1, annotation_id=1
    )
    assert len(annotations) == 1
    assert annotations[0]["segmentation"] != []


def test_coco_iscrowd_mask_exports_as_rle() -> None:
    """Multi-segment mask exports segmentation as RLE dict (iscrowd inferred as 1)."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True  # top-left component
    mask[7:9, 7:9] = True  # bottom-right component (two separate regions)

    detections = Detections(
        xyxy=np.array([[1, 1, 8, 8]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
        mask=np.array([mask]),
    )
    annotations, _ = detections_to_coco_annotations(
        detections=detections, image_id=1, annotation_id=1
    )
    assert len(annotations) == 1
    seg = annotations[0]["segmentation"]
    assert isinstance(seg, dict), "multi-segment mask must export as RLE dict, not list"
    assert "counts" in seg
    assert "size" in seg


class TestCocoExportBoxArithmetic:
    @pytest.mark.parametrize(
        ("dtype", "xyxy", "expected_bbox", "expected_area"),
        [
            pytest.param(
                "uint8", [0, 0, 200, 200], [0, 0, 200, 200], 40000, id="uint8-area"
            ),
            pytest.param(
                "int16", [0, 0, 300, 300], [0, 0, 300, 300], 90000, id="int16-area"
            ),
            pytest.param(
                "uint16", [0, 0, 300, 300], [0, 0, 300, 300], 90000, id="uint16-area"
            ),
            pytest.param(
                "int32",
                [0, 0, 65536, 65536],
                [0, 0, 65536, 65536],
                2**32,
                id="int32-area",
            ),
            pytest.param(
                "float16", [0, 0, 300, 300], [0, 0, 300, 300], 90000, id="float16-area"
            ),
            pytest.param(
                "float16",
                [-40000, 0, 40000, 1],
                [-40000, 0, 80000, 1],
                80000,
                id="float16-width",
            ),
            pytest.param(
                "float32",
                [0, 0, 2**70, 2**70],
                [0, 0, 2**70, 2**70],
                2**140,
                id="float32-area",
            ),
            pytest.param(
                "int64",
                [2**53 + 1, 2**53 + 1, 2**53 + 4, 2**53 + 6],
                [2**53 + 1, 2**53 + 1, 3, 5],
                15,
                id="int64-origin",
            ),
            pytest.param(
                "uint64",
                [2**63 + 1, 2**63 + 1, 2**63 + 4, 2**63 + 6],
                [2**63 + 1, 2**63 + 1, 3, 5],
                15,
                id="uint64-origin",
            ),
            pytest.param(
                "float32",
                [0.25, 1.5, 300.75, 302.5],
                [0.25, 1.5, 300.5, 301],
                90450.5,
                id="float32-fractional",
            ),
            pytest.param(
                "float64",
                [0.25, 1.5, 300.75, 302.5],
                [0.25, 1.5, 300.5, 301],
                90450.5,
                id="float64-fractional",
            ),
            pytest.param(
                "int16", [300, 300, 0, 0], [0, 0, 300, 300], 90000, id="reversed-int16"
            ),
        ],
    )
    def test_preserves_box_extents_and_area(
        self,
        dtype: str,
        xyxy: list[int | float],
        expected_bbox: list[int | float],
        expected_area: int | float,
    ) -> None:
        """Export finite extents and areas without losing integer origins."""
        detections = Detections(
            xyxy=np.array([xyxy], dtype=dtype), class_id=np.array([0])
        )

        annotations, _ = detections_to_coco_annotations(
            detections=detections, image_id=1, annotation_id=1
        )

        bbox = [
            value.item() if isinstance(value, np.generic) else value
            for value in annotations[0]["bbox"]
        ]
        assert bbox == expected_bbox
        assert annotations[0]["area"] == expected_area

    @pytest.mark.parametrize("dtype", ["float16", "int16"])
    def test_round_trip_keeps_finite_area(self, tmp_path: Path, dtype: str) -> None:
        """Write valid JSON and retain the box area when loading it again."""
        image_path = str(tmp_path / "image.png")
        assert cv2.imwrite(image_path, np.zeros((320, 320, 3), dtype=np.uint8))
        detections = Detections(
            xyxy=np.array([[0, 0, 300, 300]], dtype=dtype),
            class_id=np.array([0]),
        )
        dataset = DetectionDataset(
            classes=["object"],
            images=[image_path],
            annotations={image_path: detections},
        )
        annotation_path = tmp_path / "annotations.json"

        dataset.as_coco(annotations_path=str(annotation_path))
        exported = json.loads(
            annotation_path.read_text(encoding="utf-8"),
            parse_constant=lambda token: pytest.fail(
                f"Non-finite JSON constant: {token}"
            ),
        )
        restored = DetectionDataset.from_coco(
            images_directory_path=str(tmp_path),
            annotations_path=str(annotation_path),
        )

        assert exported["annotations"][0]["area"] == 90000
        restored_detections = next(iter(restored.annotations.values()))
        np.testing.assert_array_equal(restored_detections.data["area"], [90000])
        np.testing.assert_array_equal(restored_detections.xyxy, [[0, 0, 300, 300]])


class TestCocoBoxExtent:
    """Tests for how the COCO loader and exporter handle a reversed box."""

    @pytest.mark.parametrize(
        ("box_width", "box_height"),
        [
            pytest.param(-40, 40, id="negative-width"),
            pytest.param(40, -40, id="negative-height"),
            pytest.param(-40, -40, id="negative-both"),
        ],
    )
    def test_rejects_a_negative_box_extent(
        self, box_width: int, box_height: int
    ) -> None:
        """A negative extent puts x_min past x_max once corners are formed."""
        annotations = [
            mock_coco_annotation(
                annotation_id=7, bbox=(50, 50, box_width, box_height), category_id=0
            )
        ]

        with pytest.raises(
            ValueError, match=r"Invalid box extent \(.*\) in COCO annotation"
        ):
            coco_annotations_to_detections(
                image_annotations=annotations,
                resolution_wh=(100, 100),
                with_masks=False,
            )

    @pytest.mark.parametrize(
        ("box_width", "box_height", "expected_xyxy"),
        [
            pytest.param(0.0, 40.0, [50.0, 50.0, 50.0, 90.0], id="zero-width"),
            pytest.param(40.0, 0.0, [50.0, 50.0, 90.0, 50.0], id="zero-height"),
            pytest.param(
                -0.0, 40.0, [50.0, 50.0, 50.0, 90.0], id="negative-zero-width"
            ),
            pytest.param(
                40.0, -0.0, [50.0, 50.0, 90.0, 50.0], id="negative-zero-height"
            ),
        ],
    )
    def test_loads_a_zero_box_extent(
        self, box_width: float, box_height: float, expected_xyxy: list[float]
    ) -> None:
        """A zero extent is degenerate but valid, so it must keep loading."""
        annotations = [
            mock_coco_annotation(
                annotation_id=1, bbox=(50, 50, box_width, box_height), category_id=0
            )
        ]

        result = coco_annotations_to_detections(
            image_annotations=annotations, resolution_wh=(100, 100), with_masks=False
        )

        assert result.xyxy[0].tolist() == expected_xyxy

    def test_exports_a_reversed_box_with_ordered_corners(self) -> None:
        """Export must not write a box the loader would refuse to read back."""
        detections = Detections(
            xyxy=np.array([[70, 70, 30, 30]], dtype=np.float32),
            class_id=np.array([0]),
        )

        annotations, _ = detections_to_coco_annotations(
            detections=detections, image_id=1, annotation_id=1
        )

        assert [float(value) for value in annotations[0]["bbox"]] == [30, 30, 40, 40]
