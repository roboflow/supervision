from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from supervision.classification.core import Classifications
from supervision.config import CLASS_NAME_DATA_FIELD, ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections
from supervision.key_points.core import KeyPoints


class _Tensor:
    """Emulate the tensor methods used by Ultralytics adapters."""

    def __init__(self, data: np.ndarray) -> None:
        """Keep the property values returned by the model."""
        self.data = data

    def cpu(self) -> "_Tensor":
        """Return the CPU tensor stand-in."""
        return self

    def numpy(self) -> np.ndarray:
        """Expose the model property's NumPy values."""
        return self.data

    def numel(self) -> int:
        """Return the tensor's element count."""
        return self.data.size

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the tensor's dimensions."""
        return self.data.shape

    def int(self) -> "_Tensor":
        """Convert tracker IDs as torch's int method does."""
        return _Tensor(self.data.astype(np.int32))


def _property(data: np.ndarray, backend: str) -> Any:
    """Represent the same Ultralytics property as a tensor or NumPy array."""
    return data if backend == "numpy" else _Tensor(data)


def _boxes(backend: str, count: int, tracked: bool = False) -> SimpleNamespace:
    """Build Ultralytics box properties with deterministic coordinates and IDs."""
    return SimpleNamespace(
        xyxy=_property(np.tile([1, 1, 4, 4], (count, 1)).astype(np.float32), backend),
        conf=_property(np.full(count, 0.75, dtype=np.float32), backend),
        cls=_property(np.zeros(count, dtype=np.float32), backend),
        id=_property(np.arange(count, dtype=np.float32), backend) if tracked else None,
    )


class TestUltralyticsArrayResults:
    """Keep tensor and NumPy Results equivalent across the public adapters."""

    @pytest.mark.parametrize("backend", ["numpy", "tensor"])
    @pytest.mark.parametrize(
        "tracked", [pytest.param(False, id="plain"), pytest.param(True, id="tracked")]
    )
    @pytest.mark.parametrize(
        "empty", [pytest.param(False, id="nonempty"), pytest.param(True, id="empty")]
    )
    @pytest.mark.parametrize(
        "oriented", [pytest.param(False, id="boxes"), pytest.param(True, id="obb")]
    )
    def test_detection(
        self, backend: str, tracked: bool, empty: bool, oriented: bool
    ) -> None:
        """Keep coordinates, scores, classes and tracker IDs in both box routes."""
        count = 0 if empty else 2
        boxes = _boxes(backend, count, tracked)
        corners = np.tile([[1, 1], [4, 1], [4, 4], [1, 4]], (count, 1, 1))
        boxes.xyxyxyxy = _property(corners.astype(np.float32), backend)
        results = SimpleNamespace(
            boxes=boxes,
            obb=boxes if oriented else None,
            masks=None,
            names={0: "person"},
        )

        actual = Detections.from_ultralytics(results)

        np.testing.assert_array_equal(actual.xyxy, np.tile([1, 1, 4, 4], (count, 1)))
        np.testing.assert_array_equal(actual.confidence, np.full(count, 0.75))
        np.testing.assert_array_equal(actual.class_id, np.zeros(count, dtype=int))
        np.testing.assert_array_equal(
            actual.data[CLASS_NAME_DATA_FIELD], ["person"] * count
        )
        if tracked:
            np.testing.assert_array_equal(actual.tracker_id, np.arange(count))
            assert actual.tracker_id.dtype == np.int32
        else:
            assert actual.tracker_id is None
        if oriented:
            np.testing.assert_array_equal(
                actual.data[ORIENTED_BOX_COORDINATES], corners
            )

    @pytest.mark.parametrize("backend", ["numpy", "tensor"])
    @pytest.mark.parametrize(
        "with_boxes",
        [pytest.param(False, id="mask_only"), pytest.param(True, id="with_boxes")],
    )
    def test_segmentation(self, backend: str, with_boxes: bool) -> None:
        """Keep the mask extraction route usable with either property backend."""
        masks = np.zeros((2, 6, 8), dtype=np.float32)
        masks[:, 1:5, 1:5] = 1
        results = SimpleNamespace(
            boxes=_boxes(backend, 2) if with_boxes else None,
            obb=None,
            masks=SimpleNamespace(data=_property(masks, backend)),
            orig_shape=(6, 8),
            names={0: "person"},
        )

        detections = Detections.from_ultralytics(results)

        np.testing.assert_array_equal(detections.mask, masks.astype(bool))
        np.testing.assert_array_equal(detections.xyxy, [[1, 1, 4, 4]] * 2)
        np.testing.assert_array_equal(masks[:, 1:5, 1:5], 1)

    @pytest.mark.parametrize("backend", ["numpy", "tensor"])
    @pytest.mark.parametrize(
        "visibility",
        [pytest.param(False, id="xy"), pytest.param(True, id="xy_visibility")],
    )
    @pytest.mark.parametrize(
        "empty", [pytest.param(False, id="nonempty"), pytest.param(True, id="empty")]
    )
    def test_pose(self, backend: str, visibility: bool, empty: bool) -> None:
        """Keep pose coordinates and optional visibility aligned with box scores."""
        count = 0 if empty else 2
        xy = np.tile([[1, 2], [3, 4]], (count, 1, 1)).astype(np.float32)
        confidence = np.full((count, 2), 0.8, dtype=np.float32)
        results = SimpleNamespace(
            boxes=_boxes(backend, count),
            keypoints=SimpleNamespace(
                xy=_property(xy, backend),
                conf=_property(confidence, backend) if visibility else None,
            ),
            names={0: "person"},
        )

        keypoints = KeyPoints.from_ultralytics(results)

        if empty:
            assert keypoints == KeyPoints.empty()
        else:
            np.testing.assert_array_equal(keypoints.xy, xy)
            np.testing.assert_array_equal(keypoints.detection_confidence, [0.75, 0.75])
            if visibility:
                np.testing.assert_array_equal(keypoints.keypoint_confidence, confidence)
            else:
                assert keypoints.keypoint_confidence is None

    @pytest.mark.parametrize("backend", ["numpy", "tensor"])
    @pytest.mark.parametrize(
        "empty", [pytest.param(False, id="nonempty"), pytest.param(True, id="empty")]
    )
    def test_classification(self, backend: str, empty: bool) -> None:
        """Preserve classification probabilities without applying another softmax."""
        scores = (
            np.empty(0, dtype=np.float32)
            if empty
            else np.array([0.25, 0.75], dtype=np.float32)
        )
        results = SimpleNamespace(
            probs=SimpleNamespace(data=_property(scores, backend))
        )

        classifications = Classifications.from_ultralytics(results)

        np.testing.assert_array_equal(classifications.confidence, scores)
        np.testing.assert_array_equal(classifications.class_id, np.arange(len(scores)))
