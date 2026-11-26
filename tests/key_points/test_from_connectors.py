"""Tests for the `KeyPoints.from_*` model connectors."""

import numpy as np
import pytest

from supervision.config import (
    HANDEDNESS_SCORE_DATA_FIELD,
)
from supervision.key_points.annotators import EdgeAnnotator
from supervision.key_points.core import KeyPoints
from supervision.key_points.skeletons import Skeleton
from tests.helpers import (
    _create_key_points,
    _FakeMediapipeClassification,
    _FakeMediapipeHandedness,
    _FakeMediapipeHandednessCategory,
    _FakeMediapipeLandmark,
    _FakeMediapipeLandmarkWithNoneVisibility,
    _FakeMediapipeLandmarkWithoutVisibility,
    _FakeMediapipeLandmarkWithZeroVisibility,
    _FakeMediapipePose,
    _FakeMediapipeResults,
    _FakeTensor,
    _FakeUltralyticsBoxes,
    _FakeYoloNasKeyPoint,
    _FakeYoloNasKeyPointResults,
)


@pytest.mark.parametrize(
    ("inference_results", "expected_key_points"),
    [
        (
            {
                "predictions": [
                    {
                        "class_id": 1,
                        "class": "person",
                        "keypoints": [
                            {"x": 100, "y": 150, "confidence": 0.9},
                            {"x": 120, "y": 160, "confidence": 0.85},
                        ],
                    }
                ]
            },
            _create_key_points(
                xy=[[[100.0, 150.0], [120.0, 160.0]]],
                confidence=[[0.9, 0.85]],
                class_id=[1],
                data={"class_name": np.array(["person"])},
            ),
        ),
        ({"predictions": []}, KeyPoints.empty()),
    ],
)
def test_from_inference_input(inference_results, expected_key_points):
    """Test the from_inference method with valid input."""
    key_points = KeyPoints.from_inference(inference_results)
    assert key_points == expected_key_points


def test_from_inference_invalid_input():
    """Test the from_inference method with invalid input."""
    key_points = _create_key_points(
        xy=[[[0, 1], [2, 3]]], confidence=[[0.8, 0.9]], class_id=[0]
    )
    with pytest.raises(
        ValueError, match=r"from_inference\(\) operates on a single result at a time.*"
    ):
        KeyPoints.from_inference([key_points])


def _inference_keypoint(
    keypoint_id: int, xy: tuple[float, float], confidence: float
) -> dict:
    """Build one key point the way Inference serializes it, with its skeleton index."""
    return {
        "x": xy[0],
        "y": xy[1],
        "confidence": confidence,
        "class_id": keypoint_id,
        "class": f"keypoint-{keypoint_id}",
    }


def _inference_pose_result(keypoint_lists: list[list[dict]]) -> dict:
    """Wrap per-object key point lists in an Inference keypoint-detection response."""
    return {
        "image": {"width": 100, "height": 100},
        "predictions": [
            {
                "x": 50,
                "y": 50,
                "width": 20,
                "height": 40,
                "confidence": 0.9,
                "class": "person",
                "class_id": 0,
                "keypoints": keypoints,
            }
            for keypoints in keypoint_lists
        ],
    }


class TestFromInferenceOmittedKeypoints:
    """Inference leaves out key points below `keypoint_confidence`."""

    @pytest.mark.parametrize(
        ("keypoint_lists", "expected_xy", "expected_confidence"),
        [
            pytest.param(
                [
                    [
                        _inference_keypoint(0, (10, 11), 0.9),
                        _inference_keypoint(1, (20, 21), 0.8),
                        _inference_keypoint(2, (30, 31), 0.7),
                    ],
                    [
                        _inference_keypoint(0, (40, 41), 0.9),
                        _inference_keypoint(2, (60, 61), 0.6),
                    ],
                ],
                [[[10, 11], [20, 21], [30, 31]], [[40, 41], [0, 0], [60, 61]]],
                [[0.9, 0.8, 0.7], [0.9, 0.0, 0.6]],
                id="objects-with-different-keypoint-counts",
            ),
            pytest.param(
                [
                    [
                        _inference_keypoint(0, (10, 11), 0.9),
                        _inference_keypoint(1, (20, 21), 0.8),
                    ],
                    [
                        _inference_keypoint(0, (40, 41), 0.9),
                        _inference_keypoint(2, (60, 61), 0.6),
                    ],
                ],
                [[[10, 11], [20, 21], [0, 0]], [[40, 41], [0, 0], [60, 61]]],
                [[0.9, 0.8, 0.0], [0.9, 0.0, 0.6]],
                id="objects-missing-different-keypoints",
            ),
        ],
    )
    def test_places_each_keypoint_at_its_skeleton_index(
        self,
        keypoint_lists: list[list[dict]],
        expected_xy: list,
        expected_confidence: list,
    ) -> None:
        """An omitted key point leaves its own slot empty instead of shifting others."""
        result = _inference_pose_result(keypoint_lists)

        key_points = KeyPoints.from_inference(result)

        np.testing.assert_array_equal(
            key_points.xy, np.array(expected_xy, dtype=np.float32)
        )
        np.testing.assert_allclose(
            key_points.keypoint_confidence,
            np.array(expected_confidence, dtype=np.float32),
        )

    def test_omitted_keypoint_is_left_out_of_the_detection_box(self) -> None:
        """The empty slot does not stretch the box built by `as_detections`."""
        result = _inference_pose_result(
            [
                [
                    _inference_keypoint(0, (40, 41), 0.9),
                    _inference_keypoint(2, (60, 61), 0.6),
                ],
                [
                    _inference_keypoint(0, (10, 11), 0.9),
                    _inference_keypoint(1, (20, 21), 0.8),
                    _inference_keypoint(2, (30, 31), 0.7),
                ],
            ]
        )

        detections = KeyPoints.from_inference(result).as_detections()

        np.testing.assert_array_equal(
            detections.xyxy,
            np.array([[40, 41, 60, 61], [10, 11, 30, 31]], dtype=np.float32),
        )

    def test_prediction_confidence_becomes_detection_confidence(self) -> None:
        """Each prediction's score is kept as its `detection_confidence`."""
        result = _inference_pose_result(
            [
                [
                    _inference_keypoint(0, (10, 11), 0.6),
                    _inference_keypoint(2, (30, 31), 0.4),
                ],
            ]
        )

        key_points = KeyPoints.from_inference(result)

        assert key_points.detection_confidence is not None
        np.testing.assert_allclose(key_points.detection_confidence, [0.9])
        np.testing.assert_allclose(
            key_points.as_detections().confidence, [0.9], rtol=1e-6
        )

    def test_mixed_prediction_confidence_uses_keypoint_mean_for_all(self) -> None:
        """One missing object score makes the batch use keypoint means throughout."""
        result = _inference_pose_result(
            [
                [
                    _inference_keypoint(0, (10, 11), 0.6),
                    _inference_keypoint(1, (20, 21), 0.4),
                ],
                [
                    _inference_keypoint(0, (30, 31), 0.8),
                    _inference_keypoint(1, (40, 41), 0.6),
                ],
            ]
        )
        result["predictions"][1].pop("confidence")

        key_points = KeyPoints.from_inference(result)

        assert key_points.detection_confidence is None
        np.testing.assert_allclose(
            key_points.as_detections().confidence, [0.5, 0.7], rtol=1e-6
        )

    def test_null_prediction_confidence_uses_keypoint_mean_for_all(self) -> None:
        """A null object score is treated as missing across the prediction batch."""
        result = _inference_pose_result(
            [
                [
                    _inference_keypoint(0, (10, 11), 0.6),
                    _inference_keypoint(1, (20, 21), 0.4),
                ],
                [
                    _inference_keypoint(0, (30, 31), 0.8),
                    _inference_keypoint(1, (40, 41), 0.6),
                ],
            ]
        )
        result["predictions"][1]["confidence"] = None

        key_points = KeyPoints.from_inference(result)

        assert key_points.detection_confidence is None
        np.testing.assert_allclose(
            key_points.as_detections().confidence, [0.5, 0.7], rtol=1e-6
        )

    def test_objects_without_keypoints_keep_their_class(self) -> None:
        """Objects whose key points were all omitted still load, with no key points."""
        result = _inference_pose_result([[], []])

        key_points = KeyPoints.from_inference(result)

        assert key_points.xy.shape == (2, 0, 2)
        np.testing.assert_array_equal(key_points.class_id, np.array([0, 0]))

    def test_objects_without_keypoints_convert_to_empty_detections(self) -> None:
        """All-omitted Inference key points produce no detection boxes."""
        result = _inference_pose_result([[], []])

        detections = KeyPoints.from_inference(result).as_detections()

        assert len(detections) == 0
        np.testing.assert_array_equal(
            detections.xyxy, np.empty((0, 4), dtype=np.float32)
        )

    def test_empty_result_keeps_explicit_keypoint_count(self) -> None:
        """An empty frame keeps the requested skeleton dimensions."""
        result = _inference_pose_result([])

        key_points = KeyPoints.from_inference(result, num_keypoints=17)

        assert key_points.xy.shape == (0, 17, 2)
        assert key_points.keypoint_confidence is not None
        assert key_points.keypoint_confidence.shape == (0, 17)
        assert key_points.detection_confidence is not None
        assert key_points.detection_confidence.shape == (0,)

    def test_explicit_keypoint_count_keeps_trailing_slots(self) -> None:
        """Key points every object lost at the end of the skeleton keep their slots."""
        result = _inference_pose_result(
            [[_inference_keypoint(i, (i + 1, i + 2), 0.9) for i in range(15)]]
        )

        key_points = KeyPoints.from_inference(result, num_keypoints=17)

        assert key_points.xy.shape == (1, 17, 2)
        np.testing.assert_array_equal(key_points.xy[0, 15:], np.zeros((2, 2)))
        assert key_points.keypoint_confidence is not None
        np.testing.assert_array_equal(key_points.keypoint_confidence[0, 15:], [0, 0])

    def test_explicit_keypoint_count_draws_coco_skeleton(self) -> None:
        """A COCO pose missing its ankles still finds the 17-point skeleton."""
        result = _inference_pose_result(
            [[_inference_keypoint(i, (i + 1, i + 2), 0.9) for i in range(15)]]
        )
        key_points = KeyPoints.from_inference(result, num_keypoints=17)

        annotated = EdgeAnnotator().annotate(
            np.zeros((30, 30, 3), dtype=np.uint8), key_points
        )

        assert np.any(annotated)

    def test_explicit_keypoint_count_keeps_objects_without_keypoints(self) -> None:
        """Objects whose key points were all omitted get zero-filled slots."""
        result = _inference_pose_result([[], []])

        key_points = KeyPoints.from_inference(result, num_keypoints=17)

        np.testing.assert_array_equal(key_points.xy, np.zeros((2, 17, 2)))

    @pytest.mark.parametrize("num_keypoints", [0, -1, 2])
    def test_rejects_keypoint_count_that_cannot_hold_keypoints(
        self, num_keypoints: int
    ) -> None:
        """A non-positive count or one below a retained index raises."""
        result = _inference_pose_result([[_inference_keypoint(2, (1, 2), 0.9)]])

        with pytest.raises(ValueError, match="num_keypoints"):
            KeyPoints.from_inference(result, num_keypoints=num_keypoints)


class _FakeUltralyticsPoseTensor(_FakeTensor):
    """Tensor stand-in that also reports its element count like `torch.Tensor`."""

    def numel(self) -> int:
        """Return the number of elements in the wrapped array."""
        return int(self._arr.size)


class _FakeUltralyticsKeypoints:
    """Ultralytics-like `Keypoints`: `conf` is `None` without a visibility column."""

    def __init__(self, data: np.ndarray) -> None:
        """Split `(N, K, 2)` or `(N, K, 3)` key point data into `xy` and `conf`."""
        self.xy = _FakeUltralyticsPoseTensor(data[..., :2])
        self.conf = _FakeTensor(data[..., 2]) if data.shape[-1] == 3 else None


class _FakeUltralyticsPoseResults:
    """Ultralytics-like pose `Results` holding boxes, names and key points."""

    def __init__(
        self,
        keypoints: np.ndarray,
        class_id: list[int],
        box_confidence: list[float] | None = None,
    ) -> None:
        """Wrap key point data with one box, score and class id per skeleton."""
        count = len(class_id)
        conf = np.ones(count) if box_confidence is None else np.array(box_confidence)
        self.keypoints = _FakeUltralyticsKeypoints(keypoints)
        self.boxes = _FakeUltralyticsBoxes(
            xyxy=np.zeros((count, 4)),
            conf=conf,
            cls=np.array(class_id, dtype=float),
        )
        self.names = {0: "person"}


class TestFromUltralytics:
    """KeyPoints.from_ultralytics for pose models with and without visibility."""

    def test_keypoints_with_visibility_keep_their_confidence(self) -> None:
        """A `(N, K, 3)` result maps the third column to `keypoint_confidence`."""
        results = _FakeUltralyticsPoseResults(
            keypoints=np.array(
                [[[10.0, 20.0, 0.9], [30.0, 40.0, 0.4]]], dtype=np.float32
            ),
            class_id=[0],
        )

        key_points = KeyPoints.from_ultralytics(results)

        np.testing.assert_array_equal(key_points.xy, [[[10.0, 20.0], [30.0, 40.0]]])
        assert key_points.keypoint_confidence is not None
        np.testing.assert_allclose(key_points.keypoint_confidence, [[0.9, 0.4]])
        np.testing.assert_array_equal(key_points.class_id, [0])
        np.testing.assert_array_equal(key_points.data["class_name"], ["person"])

    def test_keypoints_without_visibility_load_without_confidence(self) -> None:
        """A `(N, K, 2)` result, whose `conf` is `None`, loads with no confidence."""
        results = _FakeUltralyticsPoseResults(
            keypoints=np.array(
                [[[10.0, 20.0], [30.0, 40.0]], [[50.0, 60.0], [70.0, 80.0]]],
                dtype=np.float32,
            ),
            class_id=[0, 0],
        )

        key_points = KeyPoints.from_ultralytics(results)

        np.testing.assert_array_equal(
            key_points.xy,
            [[[10.0, 20.0], [30.0, 40.0]], [[50.0, 60.0], [70.0, 80.0]]],
        )
        assert key_points.keypoint_confidence is None
        np.testing.assert_array_equal(key_points.class_id, [0, 0])
        np.testing.assert_array_equal(
            key_points.data["class_name"], ["person", "person"]
        )

    def test_box_scores_become_detection_confidence(self) -> None:
        """Each pose's box score is kept as its `detection_confidence`."""
        results = _FakeUltralyticsPoseResults(
            keypoints=np.array(
                [
                    [[10.0, 20.0, 0.9], [30.0, 40.0, 0.2]],
                    [[50.0, 60.0, 0.3], [70.0, 80.0, 0.4]],
                ],
                dtype=np.float32,
            ),
            class_id=[0, 0],
            box_confidence=[0.95, 0.6],
        )

        key_points = KeyPoints.from_ultralytics(results)

        assert key_points.detection_confidence is not None
        np.testing.assert_allclose(key_points.detection_confidence, [0.95, 0.6])
        np.testing.assert_allclose(
            key_points.as_detections().confidence, [0.95, 0.6], rtol=1e-6
        )

    @pytest.mark.parametrize("depth", [2, 3])
    def test_result_without_keypoints_is_empty(self, depth: int) -> None:
        """A result with no skeletons returns `KeyPoints.empty()`."""
        results = _FakeUltralyticsPoseResults(
            keypoints=np.zeros((0, 17, depth), dtype=np.float32), class_id=[]
        )

        key_points = KeyPoints.from_ultralytics(results)

        assert key_points == KeyPoints.empty()


class TestFromTransformers:
    """KeyPoints.from_transformers for per-image pose results."""

    def test_explicit_joint_count_keeps_trailing_slots_for_consumers(self) -> None:
        """Keep a common skeleton when the last joint is filtered in one frame."""
        filtered = [
            {
                "keypoints": np.column_stack(
                    (np.arange(16) % 6 + 1, np.arange(16) % 6 + 1)
                ),
                "scores": np.ones(16),
                "labels": np.arange(16),
            }
        ]
        full = [
            {
                "keypoints": np.ones((17, 2)),
                "scores": np.ones(17),
                "labels": np.arange(17),
            }
        ]

        first = KeyPoints.from_transformers(filtered, num_keypoints=17)
        second = KeyPoints.from_transformers(full, num_keypoints=17)
        annotated = EdgeAnnotator(edges=Skeleton.COCO.value).annotate(
            np.zeros((10, 10, 3), dtype=np.uint8), first
        )
        automatic = EdgeAnnotator().annotate(
            np.zeros((10, 10, 3), dtype=np.uint8), first
        )
        merged = KeyPoints.merge([first, second])

        assert np.any(annotated)
        assert np.any(automatic)
        assert merged.xy.shape == (2, 17, 2)
        np.testing.assert_array_equal(first.xy[0, 16], [0, 0])
        assert first.keypoint_confidence is not None
        assert first.keypoint_confidence[0, 16] == 0

    @pytest.mark.parametrize(
        ("labels", "num_keypoints", "message"),
        [
            pytest.param([2], 2, "num_keypoints", id="beyond-explicit-joint-count"),
            pytest.param(
                [0, 1, -1], 2, "non-negative", id="negative-wraps-onto-retained-joint"
            ),
            pytest.param(
                [0, 1, -1],
                None,
                "non-negative",
                id="negative-with-inferred-joint-count",
            ),
            pytest.param(
                [-3], 2, "non-negative", id="negative-below-explicit-joint-count"
            ),
            pytest.param([-1], None, "non-negative", id="only-negative-label"),
            pytest.param(
                [-3], None, "non-negative", id="only-negative-below-minus-one"
            ),
        ],
    )
    def test_rejects_out_of_range_joint_label(
        self, labels: list[int], num_keypoints: int | None, message: str
    ) -> None:
        """Reject a joint label that names no slot of the skeleton."""
        results = [
            {
                "keypoints": np.ones((len(labels), 2)),
                "scores": np.ones(len(labels)),
                "labels": np.array(labels),
            }
        ]

        with pytest.raises(ValueError, match=message):
            KeyPoints.from_transformers(results, num_keypoints=num_keypoints)

    @pytest.mark.parametrize("num_keypoints", [2, None])
    def test_rejects_repeated_joint_label(self, num_keypoints: int | None) -> None:
        """Reject a pose that names one joint slot twice instead of overwriting it."""
        results = [
            {
                "keypoints": np.array([[1.0, 1.0], [2.0, 2.0], [9.0, 9.0]]),
                "scores": np.array([0.1, 0.2, 0.9]),
                "labels": np.array([0, 1, 1]),
            }
        ]

        with pytest.raises(ValueError, match="repeat"):
            KeyPoints.from_transformers(results, num_keypoints=num_keypoints)

    def test_explicit_joint_count_keeps_fully_filtered_pose(self) -> None:
        """Keep a detected pose whose joints were all filtered out."""
        results = [
            {
                "keypoints": np.empty((0, 2)),
                "scores": np.empty(0),
                "labels": np.array([], dtype=int),
            }
        ]

        key_points = KeyPoints.from_transformers(results, num_keypoints=17)

        np.testing.assert_array_equal(key_points.xy, np.zeros((1, 17, 2)))
        np.testing.assert_array_equal(key_points.class_id, [0])

    @pytest.mark.parametrize(
        "labels",
        [
            pytest.param([[0, 2], [0, 1, 2]], id="different-counts"),
            pytest.param([[0, 2], [1, 2]], id="equal-counts-different-joints"),
            pytest.param([[], [0, 1, 2]], id="first-pose-filtered-out"),
            pytest.param([[0, 1, 2], []], id="last-pose-filtered-out"),
        ],
    )
    def test_filtered_joints_keep_their_skeleton_slots(
        self, labels: list[list[int]]
    ) -> None:
        """Use ViTPose's retained joint labels instead of shifting or stacking ragged
        poses."""
        xy = np.array(
            [[[10, 20], [30, 40], [50, 60]], [[70, 80], [90, 100], [110, 120]]],
            dtype=np.float32,
        )
        scores = np.array([[0.9, 0.8, 0.7], [0.6, 0.7, 0.8]], dtype=np.float32)
        results = [
            {
                "keypoints": _FakeTensor(xy[index, slots]),
                "scores": _FakeTensor(scores[index, slots]),
                "labels": _FakeTensor(np.array(slots, dtype=np.int64)),
            }
            for index, slots in enumerate(labels)
        ]
        expected_xy = np.zeros_like(xy)
        expected_scores = np.zeros_like(scores)
        for index, slots in enumerate(labels):
            expected_xy[index, slots] = xy[index, slots]
            expected_scores[index, slots] = scores[index, slots]

        key_points = KeyPoints.from_transformers(results)

        np.testing.assert_array_equal(key_points.xy, expected_xy)
        np.testing.assert_array_equal(key_points.keypoint_confidence, expected_scores)
        np.testing.assert_array_equal(key_points.class_id, [0, 0])

    @pytest.mark.parametrize(
        "results",
        [
            pytest.param([], id="empty-list"),
            pytest.param((), id="empty-tuple"),
            pytest.param(
                [
                    {
                        "keypoints": _FakeTensor(np.zeros((0, 2), dtype=np.float32)),
                        "scores": _FakeTensor(np.zeros((0,), dtype=np.float32)),
                    }
                ],
                id="empty-keypoints-tensor",
            ),
            pytest.param(
                [{"scores": _FakeTensor(np.array([0.9], dtype=np.float32))}],
                id="no-keypoints-key",
            ),
        ],
    )
    def test_result_without_poses_returns_empty_key_points(
        self, results: object
    ) -> None:
        """A result that holds no poses returns `KeyPoints.empty()`."""
        key_points = KeyPoints.from_transformers(results)

        assert key_points == KeyPoints.empty()

    def test_none_result_raises_type_error(self) -> None:
        """A `None` result is a caller bug, not an image without poses."""
        with pytest.raises(TypeError):
            KeyPoints.from_transformers(None)

    def test_pose_result_preserves_coordinates_and_scores(self) -> None:
        """A nonempty pose result still carries its keypoints and scores."""
        results = [
            {
                "keypoints": _FakeTensor(
                    np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
                ),
                "scores": _FakeTensor(np.array([0.9, 0.7], dtype=np.float32)),
            }
        ]

        key_points = KeyPoints.from_transformers(results)

        assert key_points.xy.shape == (1, 2, 2)
        assert key_points.xy.dtype == np.float32
        np.testing.assert_array_equal(key_points.xy, [[[10.0, 20.0], [30.0, 40.0]]])
        assert key_points.keypoint_confidence is not None
        np.testing.assert_allclose(key_points.keypoint_confidence, [[0.9, 0.7]])
        np.testing.assert_array_equal(key_points.class_id, [0])

    def test_pose_result_accepts_numpy_arrays(self) -> None:
        """Preserve support for NumPy pose coordinates and scores."""
        results = [
            {
                "keypoints": np.array([[10.0, 20.0], [30.0, 40.0]]),
                "scores": np.array([0.5, 0.25]),
            }
        ]

        key_points = KeyPoints.from_transformers(results)

        np.testing.assert_array_equal(key_points.xy, [[[10.0, 20.0], [30.0, 40.0]]])
        assert key_points.keypoint_confidence is not None
        np.testing.assert_array_equal(key_points.keypoint_confidence, [[0.5, 0.25]])

    @pytest.mark.parametrize("dtype_name", ["bfloat16", "float16", "float32"])
    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_pose_result_accepts_tensor_precisions(
        self, dtype_name: str, requires_grad: bool
    ) -> None:
        """Half-precision pose outputs load, with bfloat16 widened to float32."""
        torch = pytest.importorskip("torch")
        dtype = getattr(torch, dtype_name)
        results = [
            {
                "keypoints": torch.tensor(
                    [[10.0, 20.0], [30.0, 40.0]],
                    dtype=dtype,
                    requires_grad=requires_grad,
                ),
                "scores": torch.tensor(
                    [0.5, 0.25], dtype=dtype, requires_grad=requires_grad
                ),
            }
        ]

        key_points = KeyPoints.from_transformers(results)

        np.testing.assert_array_equal(key_points.xy, [[[10.0, 20.0], [30.0, 40.0]]])
        assert key_points.keypoint_confidence is not None
        np.testing.assert_array_equal(key_points.keypoint_confidence, [[0.5, 0.25]])
        np.testing.assert_array_equal(key_points.class_id, [0])

    def test_multiple_poses_are_stacked_with_one_shared_class_id(self) -> None:
        """Several poses stack into one `(N, K, 2)` array, all with class ID `0`."""
        results = [
            {
                "keypoints": _FakeTensor(
                    np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
                ),
                "scores": _FakeTensor(np.array([0.9, 0.7], dtype=np.float32)),
            },
            {
                "keypoints": _FakeTensor(
                    np.array([[50.0, 60.0], [70.0, 80.0]], dtype=np.float32)
                ),
                "scores": _FakeTensor(np.array([0.6, 0.5], dtype=np.float32)),
            },
        ]

        key_points = KeyPoints.from_transformers(results)

        assert key_points.xy.shape == (2, 2, 2)
        assert key_points.xy.dtype == np.float32
        np.testing.assert_array_equal(key_points.class_id, [0, 0])


class _FakeDetectron2Instances:
    """Detectron2-like `Instances` holding key points, scores and classes."""

    def __init__(
        self, keypoints: np.ndarray, scores: list[float], class_id: list[int]
    ) -> None:
        """Wrap `(N, K, 3)` key points with one score and class id per object."""
        self.pred_keypoints = _FakeTensor(keypoints)
        self.scores = _FakeTensor(np.array(scores, dtype=np.float32))
        self.pred_classes = _FakeTensor(np.array(class_id))


class TestFromDetectron2:
    """KeyPoints.from_detectron2 for keypoint R-CNN results."""

    def test_instance_scores_become_detection_confidence(self) -> None:
        """Each instance's score is kept as its `detection_confidence`."""
        instances = _FakeDetectron2Instances(
            keypoints=np.array(
                [
                    [[10.0, 20.0, 0.9], [30.0, 40.0, 0.2]],
                    [[50.0, 60.0, 0.3], [70.0, 80.0, 0.4]],
                ],
                dtype=np.float32,
            ),
            scores=[0.95, 0.6],
            class_id=[0, 0],
        )

        key_points = KeyPoints.from_detectron2({"instances": instances})

        np.testing.assert_array_equal(
            key_points.xy, [[[10.0, 20.0], [30.0, 40.0]], [[50.0, 60.0], [70.0, 80.0]]]
        )
        np.testing.assert_allclose(
            key_points.keypoint_confidence, [[0.9, 0.2], [0.3, 0.4]]
        )
        assert key_points.detection_confidence is not None
        np.testing.assert_allclose(key_points.detection_confidence, [0.95, 0.6])


@pytest.mark.parametrize(
    ("yolo_nas_results", "expected_key_points"),
    [
        (
            _FakeYoloNasKeyPointResults(
                _FakeYoloNasKeyPoint(
                    poses=[[[100.0, 150.0, 0.9], [120.0, 160.0, 0.85]]],
                    scores=[0.95],
                    labels=[1],
                ),
            ),
            _create_key_points(
                xy=[[[100.0, 150.0], [120.0, 160.0]]],
                confidence=[[0.9, 0.85]],
                class_id=[1],
                detection_confidence=[0.95],
            ),
        ),
        (
            _FakeYoloNasKeyPointResults(
                _FakeYoloNasKeyPoint(
                    poses=[],
                    scores=[],
                ),
            ),
            KeyPoints.empty(),
        ),
    ],
)
def test_from_yolo_nas_input(yolo_nas_results, expected_key_points):
    """Test the from_yolo_nas method with valid input."""
    key_points = KeyPoints.from_yolo_nas(yolo_nas_results)
    assert key_points == expected_key_points


@pytest.mark.parametrize(
    ("mediapipe_results", "resolution_wh", "expected_key_points"),
    [
        (
            _FakeMediapipeResults(
                pose_landmarks=_FakeMediapipePose(
                    landmarks=[
                        _FakeMediapipeLandmark(0.5, 0.75, 0.9),
                        _FakeMediapipeLandmark(0.6, 0.8, 0.85),
                    ]
                )
            ),
            (200, 200),
            _create_key_points(
                xy=[[[100.0, 150.0], [120.0, 160.0]]],
                confidence=[[0.9, 0.85]],
                class_id=None,
            ),
        ),
        (
            _FakeMediapipeResults(
                pose_landmarks=[
                    [
                        _FakeMediapipeLandmark(0.5, 0.75, 0.9),
                        _FakeMediapipeLandmark(0.6, 0.8, 0.85),
                    ]
                ]
            ),
            (200, 200),
            _create_key_points(
                xy=[[[100.0, 150.0], [120.0, 160.0]]],
                confidence=[[0.9, 0.85]],
                class_id=None,
            ),
        ),
        (
            _FakeMediapipeResults(
                hand_landmarks=[
                    [
                        _FakeMediapipeLandmarkWithNoneVisibility(0.1, 0.2),
                        _FakeMediapipeLandmarkWithNoneVisibility(0.3, 0.4),
                    ]
                ]
            ),
            (100, 200),
            _create_key_points(
                xy=[[[10.0, 40.0], [30.0, 80.0]]],
                confidence=[[1.0, 1.0]],
                class_id=None,
            ),
        ),  # hand_landmarks: Tasks API list-of-list; unset visibility -> forced to 1.0
        (
            _FakeMediapipeResults(
                multi_hand_landmarks=[
                    _FakeMediapipePose(
                        landmarks=[
                            _FakeMediapipeLandmarkWithZeroVisibility(0.1, 0.2),
                            _FakeMediapipeLandmarkWithZeroVisibility(0.3, 0.4),
                        ]
                    )
                ]
            ),
            (100, 200),
            _create_key_points(
                xy=[[[10.0, 40.0], [30.0, 80.0]]],
                confidence=[[1.0, 1.0]],
                class_id=None,
            ),
        ),  # multi_hand_landmarks: legacy proto2; visibility always forced to 1.0
        (
            _FakeMediapipeResults(
                face_landmarks=[
                    [
                        _FakeMediapipeLandmarkWithNoneVisibility(0.1, 0.2),
                        _FakeMediapipeLandmarkWithNoneVisibility(0.3, 0.4),
                    ]
                ]
            ),
            (100, 200),
            _create_key_points(
                xy=[[[10.0, 40.0], [30.0, 80.0]]],
                confidence=[[1.0, 1.0]],
                class_id=None,
            ),
        ),  # face_landmarks: Tasks API list-of-list; unset visibility -> forced to 1.0
        (
            _FakeMediapipeResults(
                multi_face_landmarks=[
                    _FakeMediapipePose(
                        landmarks=[
                            _FakeMediapipeLandmarkWithZeroVisibility(0.1, 0.2),
                            _FakeMediapipeLandmarkWithZeroVisibility(0.3, 0.4),
                        ]
                    )
                ]
            ),
            (100, 200),
            _create_key_points(
                xy=[[[10.0, 40.0], [30.0, 80.0]]],
                confidence=[[1.0, 1.0]],
                class_id=None,
            ),
        ),  # multi_face_landmarks: legacy proto2; visibility always forced to 1.0
    ],
)
def test_from_mediapipe_input(mediapipe_results, resolution_wh, expected_key_points):
    """Test the from_mediapipe method with valid input."""
    key_points = KeyPoints.from_mediapipe(
        mediapipe_results, resolution_wh=resolution_wh
    )
    assert key_points == expected_key_points


def test_from_mediapipe_unknown_result_type_raises() -> None:
    """Unsupported Mediapipe result objects raise a descriptive ValueError."""
    with pytest.raises(ValueError, match="Unsupported MediaPipe result"):
        KeyPoints.from_mediapipe(object(), resolution_wh=(100, 100))


def _hand_landmarks(count: int) -> list[list[_FakeMediapipeLandmarkWithoutVisibility]]:
    """Build `count` two-landmark hands for the Tasks API `hand_landmarks` shape."""
    return [
        [
            _FakeMediapipeLandmarkWithoutVisibility(0.1, 0.2),
            _FakeMediapipeLandmarkWithoutVisibility(0.3, 0.4),
        ]
        for _ in range(count)
    ]


def _legacy_hand_landmarks(count: int) -> list[_FakeMediapipePose]:
    """Build `count` two-landmark hands for the legacy `multi_hand_landmarks` shape."""
    return [_FakeMediapipePose(landmarks=hand) for hand in _hand_landmarks(count)]


class TestFromMediapipeHandedness:
    """Tests for handedness-derived `class_id` and `data` on MediaPipe hand results."""

    @pytest.mark.parametrize(
        "mediapipe_results",
        [
            pytest.param(
                _FakeMediapipeResults(
                    hand_landmarks=_hand_landmarks(2),
                    handedness=[
                        [_FakeMediapipeHandednessCategory("Left", 0.98)],
                        [_FakeMediapipeHandednessCategory("Right", 0.75)],
                    ],
                ),
                id="tasks-api",
            ),
            pytest.param(
                _FakeMediapipeResults(
                    multi_hand_landmarks=_legacy_hand_landmarks(2),
                    multi_handedness=[
                        _FakeMediapipeHandedness(
                            [_FakeMediapipeClassification("Left", 0.98)]
                        ),
                        _FakeMediapipeHandedness(
                            [_FakeMediapipeClassification("Right", 0.75)]
                        ),
                    ],
                ),
                id="legacy-solution",
            ),
        ],
    )
    def test_maps_labels_to_class_id_and_keeps_score_in_data(self, mediapipe_results):
        """Left/Right map to stable class ids and the score lands in `data`.

        Both MediaPipe generations report the same decision under different attribute
        names, so each must yield an identical `KeyPoints` regardless of which API
        produced the result.
        """
        key_points = KeyPoints.from_mediapipe(
            mediapipe_results, resolution_wh=(100, 200)
        )

        np.testing.assert_array_equal(key_points.class_id, np.array([0, 1]))
        np.testing.assert_allclose(
            key_points.data[HANDEDNESS_SCORE_DATA_FIELD], np.array([0.98, 0.75])
        )

    def test_omits_handedness_when_absent(self):
        """Hand results without handedness keep `class_id` and `data` unset.

        Older callers and partial mocks supply landmarks alone; reading handedness must
        stay optional rather than raising on the missing attribute.
        """
        mediapipe_results = _FakeMediapipeResults(hand_landmarks=_hand_landmarks(1))

        key_points = KeyPoints.from_mediapipe(
            mediapipe_results, resolution_wh=(100, 200)
        )

        assert key_points.class_id is None
        assert HANDEDNESS_SCORE_DATA_FIELD not in key_points.data

    @pytest.mark.parametrize(
        "mediapipe_results",
        [
            pytest.param(
                _FakeMediapipeResults(
                    hand_landmarks=_hand_landmarks(2),
                    handedness=[[_FakeMediapipeHandednessCategory("Left", 0.98)]],
                ),
                id="fewer-handedness-entries-than-hands",
            ),
            pytest.param(
                _FakeMediapipeResults(
                    hand_landmarks=_hand_landmarks(1),
                    handedness=[[_FakeMediapipeHandednessCategory("Middle", 0.98)]],
                ),
                id="unrecognised-label",
            ),
        ],
    )
    def test_drops_handedness_when_it_cannot_align_with_hands(self, mediapipe_results):
        """Misaligned or unrecognised handedness is dropped whole, never guessed.

        A per-hand fallback would silently shift every later `class_id` by one, so the
        contract is all-or-nothing: emit both arrays or neither.
        """
        key_points = KeyPoints.from_mediapipe(
            mediapipe_results, resolution_wh=(100, 200)
        )

        assert key_points.class_id is None
        assert HANDEDNESS_SCORE_DATA_FIELD not in key_points.data

    def test_leaves_pose_results_untouched(self):
        """Pose results keep their existing `class_id`/`data` behaviour.

        Handedness handling is additive for hand branches only; this pins that the pose
        path did not inherit an empty-but-present data field or a class id.
        """
        mediapipe_results = _FakeMediapipeResults(
            pose_landmarks=_FakeMediapipePose(
                landmarks=[_FakeMediapipeLandmark(0.5, 0.75, 0.9)]
            )
        )

        key_points = KeyPoints.from_mediapipe(
            mediapipe_results, resolution_wh=(200, 200)
        )

        assert key_points.class_id is None
        assert HANDEDNESS_SCORE_DATA_FIELD not in key_points.data
