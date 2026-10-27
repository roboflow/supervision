import math
from typing import Any

import numpy as np
import pytest

import supervision as sv
from supervision.config import (
    CLASS_HEADING_METADATA_FIELD,
    HEADING_DATA_FIELD,
    ORIENTED_BOX_COORDINATES,
)


def _response() -> dict[str, Any]:
    """Build same-footprint detections with opposite observed semantic fronts."""
    return {
        "task_type": "oriented-object-detection",
        "class_heading": {"plane": True, "stone": False},
        "image": {"width": 101, "height": 79},
        "predictions": [
            {
                "obb": [40.25, 30.75, 12.5, 25.75, -2.9],
                "class": "plane",
                "class_id": 7,
                "confidence": 0.6,
                "heading": True,
                "detection_id": "first",
            },
            {
                "obb": [40.25, 30.75, 12.5, 25.75, -2.9 + math.pi],
                "class": "plane",
                "class_id": 7,
                "confidence": 0.9,
                "heading": True,
                "detection_id": "second",
            },
        ],
    }


class TestNativeOBBInference:
    @pytest.mark.parametrize("angle", [0.0, math.pi / 2, -math.pi / 2, math.pi])
    def test_keeps_ordered_shape_and_front_axis(self, angle: float) -> None:
        """Ordered sides stay intact and the first edge points along theta."""
        response = _response()
        response["predictions"][0]["obb"][4] = angle

        detections = sv.Detections.from_inference(response)

        corners = detections.data[ORIENTED_BOX_COORDINATES]
        assert isinstance(corners, np.ndarray)
        edge = corners[0, 1] - corners[0, 0]
        np.testing.assert_allclose(
            edge, [12.5 * math.cos(angle), 12.5 * math.sin(angle)], atol=1e-12
        )
        assert corners.dtype == np.float64
        np.testing.assert_array_equal(detections.data[HEADING_DATA_FIELD], [True, True])
        assert (
            detections.metadata[CLASS_HEADING_METADATA_FIELD]
            == response["class_heading"]
        )

    @pytest.mark.parametrize("field", ["heading", "obb", "detection_id"])
    def test_rejects_missing_native_field(self, field: str) -> None:
        """A required native field cannot silently become an ordinary detection."""
        response = _response()
        del response["predictions"][0][field]

        with pytest.raises(ValueError, match=r"native OBB|Native OBB"):
            sv.Detections.from_inference(response)

    def test_preserves_empty_class_contract(self) -> None:
        """An empty image keeps frozen capabilities and an empty corner array."""
        response = _response()
        response["predictions"] = []

        detections = sv.Detections.from_inference(response)

        corners = detections.data[ORIENTED_BOX_COORDINATES]
        assert isinstance(corners, np.ndarray)
        assert corners.shape == (0, 4, 2)
        assert (
            detections.metadata[CLASS_HEADING_METADATA_FIELD]
            == response["class_heading"]
        )

    def test_nms_retains_the_selected_observed_front(self) -> None:
        """Rotated NMS selects one row with its original direction and identity."""
        source = sv.Detections.from_inference(_response())

        result = source.with_nms(threshold=0.5)

        np.testing.assert_array_equal(
            result.data[ORIENTED_BOX_COORDINATES],
            source.data[ORIENTED_BOX_COORDINATES][[1]],
        )
        np.testing.assert_array_equal(result.data["detection_id"], ["second"])

    def test_nmm_retains_exact_winner_instead_of_refitting(self) -> None:
        """Conflicting fronts select the highest-confidence observed rectangle."""
        response = _response()
        response["predictions"][1]["obb"][0] += 1.0
        source = sv.Detections.from_inference(response)

        result = source.with_nmm(threshold=0.5)

        np.testing.assert_array_equal(
            result.data[ORIENTED_BOX_COORDINATES],
            source.data[ORIENTED_BOX_COORDINATES][[1]],
        )
        np.testing.assert_array_equal(result.data["detection_id"], ["second"])
        np.testing.assert_array_equal(result.confidence, [0.9])

    def test_nms_uses_rotated_overlap_instead_of_enclosing_aabb(self) -> None:
        """Thin crossing rectangles share AABBs but not enough rotated area."""
        response = _response()
        response["predictions"][0]["obb"] = [40, 30, 30, 2, math.pi / 4]
        response["predictions"][1]["obb"] = [40, 30, 30, 2, -math.pi / 4]
        source = sv.Detections.from_inference(response)

        result = source.with_nms(threshold=0.5)

        assert len(result) == 2


class TestOrdinaryInferenceCompatibility:
    def test_unmarked_obb_extension_keeps_ordinary_detection_contract(self) -> None:
        """An unrelated extension cannot change the existing task discriminator."""
        response = {
            "image": {"width": 100, "height": 100},
            "predictions": [
                {
                    "x": 20,
                    "y": 30,
                    "width": 10,
                    "height": 8,
                    "confidence": 0.9,
                    "class_id": 0,
                    "class": "object",
                    "obb": "caller-owned extension",
                }
            ],
        }

        detections = sv.Detections.from_inference(response)

        np.testing.assert_array_equal(detections.xyxy, [[15, 26, 25, 34]])
        assert ORIENTED_BOX_COORDINATES not in detections.data
