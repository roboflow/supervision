import math
from typing import Any

import numpy as np
import pytest

import supervision as sv
from supervision.config import (
    CLASS_HEADING_METADATA_FIELD,
    DETECTION_ID_DATA_FIELD,
    HEADING_DATA_FIELD,
    IMAGE_DIMENSIONS_METADATA_FIELD,
    NATIVE_OBB_DATA_FIELD,
    ORIENTED_BOX_COORDINATES,
    SOURCE_IMAGE_METADATA_FIELD,
)
from supervision.detection.utils.iou_and_nms import OverlapFilter
from supervision.utils.internal import SupervisionWarnings


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
    def test_empty_sliced_native_result_retains_full_image_contract(self) -> None:
        """An all-negative tiled image retains native schema and frozen class policy."""
        image = np.zeros((70, 100, 3), dtype=np.uint8)

        def predict(tile: np.ndarray) -> sv.Detections:
            """Return an empty native response carrying its source tile."""
            response = _response()
            response["image"] = {"height": tile.shape[0], "width": tile.shape[1]}
            response["predictions"] = []
            result = sv.Detections.from_inference(response)
            result.metadata[SOURCE_IMAGE_METADATA_FIELD] = tile
            return result

        slicer = sv.InferenceSlicer(callback=predict, slice_wh=64, overlap_wh=0)

        result = slicer(image)

        assert result.is_empty()
        assert result.metadata[NATIVE_OBB_DATA_FIELD] is True
        assert result.metadata[IMAGE_DIMENSIONS_METADATA_FIELD] == [70, 100]
        assert result.metadata[CLASS_HEADING_METADATA_FIELD] == {
            "plane": True,
            "stone": False,
        }
        assert result.metadata[SOURCE_IMAGE_METADATA_FIELD] is image
        corners = result.data[ORIENTED_BOX_COORDINATES]
        assert isinstance(corners, np.ndarray)
        assert corners.shape == (0, 4, 2)
        assert corners.dtype == np.float64
        assert result.data[HEADING_DATA_FIELD].dtype == bool
        assert result.data[DETECTION_ID_DATA_FIELD].shape == (0,)

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize(
        ("image_hw", "offsets"),
        [
            pytest.param((64, 128), [[0, 0], [64, 0]], id="equal-tiles"),
            pytest.param(
                (70, 100), [[0, 0], [64, 0], [0, 64], [64, 64]], id="clipped-tiles"
            ),
        ],
    )
    def test_sliced_native_geometry_and_dimensions_share_full_image_frame(
        self, image_hw: tuple[int, int], offsets: list[list[int]], batch_size: int
    ) -> None:
        """Tiled native results retain their ordered front and full-image dimensions."""
        image = np.zeros((*image_hw, 3), dtype=np.uint8)
        tile_results: list[sv.Detections] = []

        def predict(tile: np.ndarray) -> sv.Detections:
            """Parse a native model response in the tile's coordinate frame."""
            response = _response()
            response["image"] = {"height": tile.shape[0], "width": tile.shape[1]}
            response["predictions"] = response["predictions"][:1]
            response["predictions"][0]["obb"] = [3.25, 3, 4, 2, math.pi / 2]
            result = sv.Detections.from_inference(response)
            tile_results.append(result)
            return result

        def predict_batch(tiles: list[np.ndarray]) -> list[sv.Detections]:
            """Return native tile results through the public batch callback."""
            return [predict(tile) for tile in tiles]

        callback = predict if batch_size == 1 else predict_batch
        slicer = sv.InferenceSlicer(
            callback=callback,
            slice_wh=64,
            overlap_wh=0,
            batch_size=batch_size,
            thread_workers=2,
            overlap_filter=OverlapFilter.NONE,
        )

        with pytest.warns(SupervisionWarnings, match="oriented bounding boxes"):
            result = slicer(image)

        expected_corners = np.array([[4.25, 1], [4.25, 5], [2.25, 5], [2.25, 1]])
        np.testing.assert_allclose(
            result.data[ORIENTED_BOX_COORDINATES],
            expected_corners[None] + np.array(offsets)[:, None],
            atol=1e-12,
        )
        assert result.metadata[IMAGE_DIMENSIONS_METADATA_FIELD] == list(image_hw)
        assert result.metadata[CLASS_HEADING_METADATA_FIELD] == {
            "plane": True,
            "stone": False,
        }
        np.testing.assert_array_equal(result.data[HEADING_DATA_FIELD], True)
        np.testing.assert_array_equal(result.data[DETECTION_ID_DATA_FIELD], "first")
        assert tile_results[0].metadata[IMAGE_DIMENSIONS_METADATA_FIELD] == [64, 64]

    @pytest.mark.parametrize("heading", [False, True])
    @pytest.mark.parametrize("angle", [0.0, math.pi / 2, -math.pi / 2, math.pi])
    def test_keeps_ordered_shape_and_front_axis(
        self, angle: float, heading: bool
    ) -> None:
        """Ordered axes and explicit directed or undirected capabilities stay intact."""
        response = _response()
        response["predictions"][0]["obb"][4] = angle
        response["class_heading"]["plane"] = heading
        for prediction in response["predictions"]:
            prediction["heading"] = heading

        detections = sv.Detections.from_inference(response)

        corners = detections.data[ORIENTED_BOX_COORDINATES]
        assert isinstance(corners, np.ndarray)
        edge = corners[0, 1] - corners[0, 0]
        np.testing.assert_allclose(
            edge, [12.5 * math.cos(angle), 12.5 * math.sin(angle)], atol=1e-12
        )
        side = corners[0, 2] - corners[0, 1]
        np.testing.assert_allclose(
            side, [-25.75 * math.sin(angle), 25.75 * math.cos(angle)], atol=1e-12
        )
        assert corners.dtype == np.float64
        np.testing.assert_array_equal(
            detections.data[HEADING_DATA_FIELD], [heading, heading]
        )
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

    @pytest.mark.parametrize(
        "capabilities",
        [
            None,
            pytest.param([True], id="non-mapping"),
            pytest.param({"plane": True, "stone": 1}, id="non-boolean-capability"),
            pytest.param({"stone": False}, id="missing-predicted-class"),
            pytest.param({"plane": False}, id="contradictory-heading"),
        ],
    )
    def test_rejects_invalid_frozen_capabilities(self, capabilities: Any) -> None:
        """Native predictions require an explicit, matching boolean class policy."""
        response = _response()
        response["class_heading"] = capabilities

        with pytest.raises(ValueError, match=r"frozen class|class capability"):
            sv.Detections.from_inference(response)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            pytest.param(
                "obb", [40, 30, 0, 20, 0], "positive ordered sides", id="zero-width"
            ),
            pytest.param(
                "obb",
                [40, 30, 12, -20, 0],
                "positive ordered sides",
                id="negative-height",
            ),
            pytest.param(
                "obb", [math.nan, 30, 12, 20, 0], "must be finite", id="nan-center"
            ),
            pytest.param(
                "obb", [40, 30, 12, 20, math.inf], "must be finite", id="infinite-angle"
            ),
            pytest.param(
                "obb", [40, 30, 12, 20], r"require \[cx", id="incomplete-tuple"
            ),
            pytest.param(
                "obb",
                [40, 30, 12, 20, 0] * 2,
                "positive ordered sides",
                id="extra-tuple",
            ),
            pytest.param(
                "confidence", math.nan, "confidence must be finite", id="nan-confidence"
            ),
            pytest.param(
                "confidence",
                math.inf,
                "confidence must be finite",
                id="infinite-confidence",
            ),
            pytest.param(
                "confidence",
                -0.1,
                "confidence must be finite",
                id="negative-confidence",
            ),
            pytest.param(
                "confidence",
                1.1,
                "confidence must be finite",
                id="confidence-above-one",
            ),
            pytest.param("heading", 1, "class capability", id="non-boolean-heading"),
            pytest.param(
                "detection_id", "", "stable detection_id", id="empty-identity"
            ),
        ],
    )
    def test_rejects_malformed_native_prediction(
        self, field: str, value: Any, message: str
    ) -> None:
        """Malformed rows fail at the adapter instead of losing native semantics."""
        response = _response()
        response["predictions"][0][field] = value
        if field == "obb":
            response["predictions"] = response["predictions"][:1]

        with pytest.raises(ValueError, match=message):
            sv.Detections.from_inference(response)

    def test_preserves_empty_class_contract(self) -> None:
        """An empty image keeps frozen capabilities and an empty corner array."""
        response = _response()
        response["predictions"] = []

        detections = sv.Detections.from_inference(response)

        corners = detections.data[ORIENTED_BOX_COORDINATES]
        assert isinstance(corners, np.ndarray)
        assert corners.shape == (0, 4, 2)
        assert detections.xyxy.shape == (0, 4)
        assert detections.xyxy.dtype == np.float64
        assert detections.confidence is not None
        assert detections.confidence.shape == (0,)
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

    @pytest.mark.parametrize(
        ("state", "message"),
        [
            pytest.param(
                "mixed", "cannot mix ordinary", id="mixed-native-and-ordinary"
            ),
            pytest.param(
                "corners", "requires ordered corners and heading", id="missing-corners"
            ),
            pytest.param(
                "heading", "requires ordered corners and heading", id="missing-heading"
            ),
            pytest.param("classes", "requires class identifiers", id="missing-classes"),
            pytest.param(
                "conflicting",
                "Conflicting native OBB class heading",
                id="conflicting-class-policy",
            ),
        ],
    )
    def test_nmm_rejects_inconsistent_mutable_native_container(
        self, state: str, message: str
    ) -> None:
        """Public container edits cannot merge mixed or incomplete native semantics."""
        source = sv.Detections.from_inference(_response())
        if state == "mixed":
            source.data[NATIVE_OBB_DATA_FIELD] = np.array([True, False])
        elif state == "corners":
            del source.data[ORIENTED_BOX_COORDINATES]
        elif state == "heading":
            del source.data[HEADING_DATA_FIELD]
        elif state == "classes":
            source.class_id = None
        else:
            source.data[HEADING_DATA_FIELD] = np.array([True, False])

        with pytest.raises(ValueError, match=message):
            source.with_nmm(threshold=0.5, class_agnostic=True)

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
