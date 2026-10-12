from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from faster_coco_eval import COCO, COCOeval_faster

from supervision.detection.utils.iou_and_nms import (
    _COCO_KEYPOINT_SIGMAS,
    _keypoint_oks_batch,
)
from supervision.key_points.core import KeyPoints
from supervision.metrics.keypoint_mean_average_precision import (
    KeyPointMeanAveragePrecision,
    KeyPointMeanAveragePrecisionResult,
    _KeyPointCOCOEvaluator,
)
from supervision.metrics.mean_average_precision import EvaluationDataset

PERSON_TEMPLATE = np.array(
    [
        [0.50, 0.08],
        [0.46, 0.05],
        [0.54, 0.05],
        [0.41, 0.07],
        [0.59, 0.07],
        [0.33, 0.22],
        [0.67, 0.22],
        [0.25, 0.40],
        [0.75, 0.40],
        [0.20, 0.55],
        [0.80, 0.55],
        [0.38, 0.55],
        [0.62, 0.55],
        [0.37, 0.77],
        [0.63, 0.77],
        [0.36, 0.98],
        [0.64, 0.98],
    ]
)


# Parity with pycocotools holds to about 1e-7, not to float64 precision: the shared
# accumulator stores precision as float32, which leaves up to ~1e-8 of drift here.
PARITY_TOLERANCE = 1e-7

# pycocotools 2.0.11 results for `_make_synthetic_pose_images()`. pycocotools is not a
# dependency, so regenerate them with
# `uv run --with pycocotools python -m tests.metrics._generate_keypoint_map_parity`
# after any change to the synthetic data, such as its RNG call order.
EXPECTED_STATS = [
    0.2508697518103459,
    0.3701084394153701,
    0.2701457645764576,
    0.20684818481848183,
    0.2680997426665743,
]
EXPECTED_AP_PER_CLASS = np.array(
    [
        [
            0.231966053748,
            0.231966053748,
            0.177274870344,
            0.177274870344,
            0.177274870344,
            0.159653465347,
            0.159653465347,
            0.136633663366,
            0.136633663366,
            0.102056359482,
        ],
        [
            0.508250825083,
            0.508250825083,
            0.508250825083,
            0.46204620462,
            0.380638063806,
            0.380638063806,
            0.380638063806,
            0.158415841584,
            0.039878987899,
            0.0,
        ],
    ]
)


@dataclass
class SyntheticPoseImage:
    """One image of synthetic pose targets and predictions."""

    targets: KeyPoints
    predictions: KeyPoints


def _random_person(
    rng: np.random.Generator, height: float, origin: npt.NDArray[np.float64]
) -> npt.NDArray[np.float32]:
    """Draw a 17-point person of the given height with its box at `origin`."""
    jitter = rng.normal(0.0, 0.02, size=PERSON_TEMPLATE.shape)
    xy = (PERSON_TEMPLATE + jitter) * np.array([0.45 * height, height]) + origin
    return xy.astype(np.float32)


def _make_synthetic_pose_images(seed: int = 7) -> list[SyntheticPoseImage]:
    """Build COCO-like keypoint data covering the cases COCOeval handles.

    Images hold 0 to 6 people of two classes and heights from 30 to 300 px, so targets
    span the small, medium and large buckets. Targets carry a segmentation-like area and
    some invisible points, and one has none visible. Most targets get a prediction whose
    noise is scaled by each keypoint's sigma and the target area, at levels that spread
    OKS across all thresholds. Some targets are missed, false positives are added, one
    image holds more than 20 predictions of a class, and scores are rounded so that many
    of them tie.
    """
    rng = np.random.default_rng(seed)
    images: list[SyntheticPoseImage] = []
    for image_index in range(10):
        num_targets = [3, 0, 6, 1, 4, 2, 5, 3, 2, 4][image_index]
        target_xy, target_visible, target_class, target_area = [], [], [], []
        pred_xy, pred_class, pred_score = [], [], []
        for _ in range(num_targets):
            height = float(rng.uniform(30, 300))
            origin = rng.uniform(0, 600, size=2)
            xy = _random_person(rng, height, origin)
            visible = rng.random(17) < 0.8
            class_id = int(rng.integers(0, 2))
            # Segmentation area is a fraction of the person's box area.
            area = float(0.45 * height * height * rng.uniform(0.5, 0.7))
            target_xy.append(xy)
            target_visible.append(visible)
            target_class.append(class_id)
            target_area.append(area)

            if rng.random() < 0.2:
                continue
            noise_level = rng.choice([0.2, 0.5, 1.0, 1.5, 2.5, 4.0])
            scale = noise_level * _COCO_KEYPOINT_SIGMAS[:, None] * np.sqrt(area)
            noisy = xy + rng.normal(0.0, 1.0, size=xy.shape) * scale
            # Unlabelled target points must not influence OKS.
            noisy[~visible] = rng.uniform(0, 900, size=(int((~visible).sum()), 2))
            pred_xy.append(noisy.astype(np.float32))
            pred_class.append(class_id)
            pred_score.append(round(float(rng.uniform(0.1, 1.0)), 1))

        num_false_positives = 22 if image_index == 6 else int(rng.integers(0, 3))
        for _ in range(num_false_positives):
            height = float(rng.uniform(30, 300))
            pred_xy.append(_random_person(rng, height, rng.uniform(0, 600, size=2)))
            pred_class.append(0 if image_index == 6 else int(rng.integers(0, 2)))
            pred_score.append(round(float(rng.uniform(0.1, 1.0)), 1))

        if image_index == 4:
            # A target with no labelled keypoint, far from every prediction.
            target_xy.append(_random_person(rng, 120.0, np.array([5000.0, 5000.0])))
            target_visible.append(np.zeros(17, dtype=bool))
            target_class.append(0)
            target_area.append(3000.0)

        targets = (
            KeyPoints(
                xy=np.array(target_xy, dtype=np.float32),
                class_id=np.array(target_class),
                visible=np.array(target_visible),
                data={"area": np.array(target_area)},
            )
            if target_xy
            else KeyPoints.empty()
        )
        predictions = (
            KeyPoints(
                xy=np.array(pred_xy, dtype=np.float32),
                class_id=np.array(pred_class),
                detection_confidence=np.array(pred_score, dtype=np.float32),
            )
            if pred_xy
            else KeyPoints.empty()
        )
        images.append(SyntheticPoseImage(targets=targets, predictions=predictions))
    return images


def _flat_keypoints(xy: np.ndarray, visibility: np.ndarray) -> list[float]:
    """Flatten `(K, 2)` coordinates and `(K,)` flags into COCO keypoints."""
    return [float(value) for (x, y), v in zip(xy, visibility) for value in (x, y, v)]


def _keypoints_span(xy: np.ndarray) -> list[float]:
    """Return the `[x, y, width, height]` box spanning `(K, 2)` keypoints."""
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def _to_coco(
    images: list[SyntheticPoseImage],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Convert synthetic pose data to COCO ground truth and result dictionaries."""
    annotations: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for index, image in enumerate(images):
        image_id = index + 1
        targets, predictions = image.targets, image.predictions
        for target_index in range(len(targets)):
            xy = targets.xy[target_index].astype(np.float64)
            visible = np.asarray(targets.visible)[target_index]
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": int(targets.class_id[target_index]),
                    "iscrowd": 0,
                    "area": float(targets.data["area"][target_index]),
                    "bbox": _keypoints_span(xy),
                    "num_keypoints": int(visible.sum()),
                    "keypoints": _flat_keypoints(xy, 2 * visible.astype(int)),
                }
            )
        for pred_index in range(len(predictions)):
            xy = predictions.xy[pred_index].astype(np.float64)
            results.append(
                {
                    "image_id": image_id,
                    "category_id": int(predictions.class_id[pred_index]),
                    "score": float(predictions.detection_confidence[pred_index]),
                    "keypoints": _flat_keypoints(xy, np.ones(len(xy), dtype=int)),
                }
            )
    categories = [{"id": class_id, "name": str(class_id)} for class_id in (0, 1)]
    dataset = {
        "images": [{"id": index + 1} for index in range(len(images))],
        "annotations": annotations,
        "categories": categories,
    }
    return dataset, results


class TestKeyPointOksBatch:
    """Pairwise OKS between target and detected keypoint sets, and its input checks."""

    def test_identical_keypoints_have_oks_one(self) -> None:
        """A detection placed exactly on the target scores OKS 1."""
        keypoints = np.array([[[0, 0], [10, 0], [5, 10]]], dtype=np.float32)

        oks = _keypoint_oks_batch(
            keypoints, keypoints, area_true=np.array([50.0]), sigmas=[0.1, 0.1, 0.1]
        )

        assert oks == pytest.approx(np.array([[1.0]]))

    def test_matches_coco_formula(self) -> None:
        """OKS averages exp(-d^2 / (2 * area * (2 * sigma)^2)) over visible points."""
        keypoints_true = np.array([[[0, 0], [10, 0], [5, 10]]], dtype=np.float32)
        keypoints_detection = np.array([[[3, 4], [10, 0], [100, 100]]])
        area, sigma = 400.0, 0.5
        visible = np.array([[True, True, False]])

        oks = _keypoint_oks_batch(
            keypoints_true,
            keypoints_detection,
            area_true=np.array([area]),
            sigmas=[sigma] * 3,
            visible_true=visible,
        )

        expected = (np.exp(-25.0 / (2 * area * (2 * sigma) ** 2)) + 1.0) / 2
        assert oks == pytest.approx(np.array([[expected]]))

    def test_non_finite_visible_keypoint_contributes_zero(self) -> None:
        """A non-finite visible point scores zero while finite points still count."""
        keypoints_true = np.array([[[0, 0], [10, 0]]], dtype=np.float32)
        keypoints_detection = np.array([[[np.nan, 0], [10, 0]]], dtype=np.float32)

        oks = _keypoint_oks_batch(
            keypoints_true,
            keypoints_detection,
            area_true=np.array([50.0]),
            sigmas=[0.1, 0.1],
            visible_true=np.ones((1, 2), dtype=bool),
        )

        assert oks == pytest.approx(np.array([[0.5]]))

    def test_target_without_visible_keypoints_has_zero_oks(self) -> None:
        """A target with every keypoint unlabelled matches nothing."""
        keypoints = np.array([[[0, 0], [10, 0], [5, 10]]], dtype=np.float32)

        oks = _keypoint_oks_batch(
            keypoints,
            keypoints,
            area_true=np.array([50.0]),
            sigmas=[0.1, 0.1, 0.1],
            visible_true=np.zeros((1, 3), dtype=bool),
        )

        assert oks == pytest.approx(np.array([[0.0]]))

    @pytest.mark.parametrize(
        ("detection_xy", "expected_squared_distances"),
        [
            pytest.param(
                [[-9, 5], [19, -9], [5, 19]], [0.0, 0.0, 0.0], id="inside-expanded-box"
            ),
            pytest.param(
                [[-13, 5], [5, 24], [5, 5]], [9.0, 16.0, 0.0], id="outside-expanded-box"
            ),
        ],
    )
    def test_target_without_visible_keypoints_uses_expanded_box(
        self,
        detection_xy: list[list[float]],
        expected_squared_distances: list[float],
    ) -> None:
        """Without visible points, distance is measured to the expanded box.

        The box `(0, 0, 10, 10)` expands to `[-10, 20] x [-10, 20]`, as in
        `pycocotools` `computeOks`, and the mean runs over all keypoints.
        """
        keypoints_true = np.zeros((1, 3, 2))
        area, sigma = 50.0, 0.1
        distances = np.array(expected_squared_distances)
        expected = np.exp(-distances / (2 * area * (2 * sigma) ** 2)).mean()

        oks = _keypoint_oks_batch(
            keypoints_true,
            np.array([detection_xy], dtype=np.float64),
            area_true=np.array([area]),
            sigmas=[sigma] * 3,
            visible_true=np.zeros((1, 3), dtype=bool),
            xyxy_true=np.array([[0.0, 0.0, 10.0, 10.0]]),
        )

        assert oks == pytest.approx(np.array([[expected]]))

    def test_box_is_unused_for_target_with_visible_keypoints(self) -> None:
        """A target with visible points keeps the keypoint OKS despite a box."""
        keypoints = np.array([[[0, 0], [10, 0], [5, 10]]], dtype=np.float64)
        detection = keypoints + 3.0

        with_box = _keypoint_oks_batch(
            keypoints,
            detection,
            area_true=np.array([50.0]),
            sigmas=[0.1, 0.1, 0.1],
            xyxy_true=np.array([[0.0, 0.0, 10.0, 10.0]]),
        )
        without_box = _keypoint_oks_batch(
            keypoints, detection, area_true=np.array([50.0]), sigmas=[0.1, 0.1, 0.1]
        )

        assert with_box == pytest.approx(without_box)

    def test_raises_for_invalid_box_shape(self) -> None:
        """`xyxy_true` must hold one box per target."""
        keypoints = np.zeros((1, 3, 2))

        with pytest.raises(ValueError, match="xyxy_true"):
            _keypoint_oks_batch(
                keypoints,
                keypoints,
                area_true=np.array([1.0]),
                sigmas=[0.1, 0.1, 0.1],
                xyxy_true=np.zeros((2, 4)),
            )

    @pytest.mark.parametrize(
        ("num_true", "num_detection"),
        [
            pytest.param(0, 2, id="no-targets"),
            pytest.param(2, 0, id="no-detections"),
            pytest.param(0, 0, id="both-empty"),
        ],
    )
    def test_empty_input_gives_empty_matrix(
        self, num_true: int, num_detection: int
    ) -> None:
        """Empty inputs give an `(N, M)` matrix without needing sigmas."""
        oks = _keypoint_oks_batch(
            np.zeros((num_true, 5, 2)),
            np.zeros((num_detection, 5, 2)),
            area_true=np.ones(num_true),
        )

        assert oks.shape == (num_true, num_detection)

    def test_defaults_to_coco_sigmas_for_17_keypoints(self) -> None:
        """Without sigmas, 17-point skeletons use the COCO preset."""
        keypoints_true = np.zeros((1, 17, 2))
        keypoints_detection = np.ones((1, 17, 2))
        area = np.array([100.0])

        oks = _keypoint_oks_batch(keypoints_true, keypoints_detection, area)

        expected = _keypoint_oks_batch(
            keypoints_true, keypoints_detection, area, sigmas=_COCO_KEYPOINT_SIGMAS
        )
        assert oks == pytest.approx(expected)

    def test_raises_without_sigmas_for_non_coco_skeleton(self) -> None:
        """Skeletons that are not 17 points long need explicit sigmas."""
        keypoints = np.zeros((1, 5, 2))

        with pytest.raises(ValueError, match="sigma"):
            _keypoint_oks_batch(keypoints, keypoints, area_true=np.array([1.0]))

    @pytest.mark.parametrize(
        "sigmas",
        [
            pytest.param([0.1, 0.1], id="wrong-length"),
            pytest.param([0.1, 0.0, 0.1], id="non-positive"),
            pytest.param([0.1, np.inf, 0.1], id="infinite"),
            pytest.param([0.1, np.nan, 0.1], id="nan"),
        ],
    )
    def test_raises_for_invalid_sigmas(self, sigmas: list[float]) -> None:
        """Sigmas must hold one positive, finite value per keypoint."""
        keypoints = np.zeros((1, 3, 2))

        with pytest.raises(ValueError, match="sigmas"):
            _keypoint_oks_batch(
                keypoints, keypoints, area_true=np.array([1.0]), sigmas=sigmas
            )

    def test_raises_for_mismatched_keypoint_counts(self) -> None:
        """Targets and detections must share one skeleton."""
        with pytest.raises(ValueError, match="shapes"):
            _keypoint_oks_batch(
                np.zeros((1, 3, 2)),
                np.zeros((1, 4, 2)),
                area_true=np.array([1.0]),
                sigmas=[0.1, 0.1, 0.1],
            )

    def test_raises_for_invalid_area_shape(self) -> None:
        """`area_true` must hold one area per target."""
        keypoints = np.zeros((1, 3, 2))

        with pytest.raises(ValueError, match="area_true"):
            _keypoint_oks_batch(
                keypoints, keypoints, area_true=np.ones(2), sigmas=[0.1, 0.1, 0.1]
            )

    def test_raises_for_invalid_visible_shape(self) -> None:
        """`visible_true` must hold one flag per target keypoint."""
        keypoints = np.zeros((1, 3, 2))

        with pytest.raises(ValueError, match="visible_true"):
            _keypoint_oks_batch(
                keypoints,
                keypoints,
                area_true=np.array([1.0]),
                sigmas=[0.1, 0.1, 0.1],
                visible_true=np.ones((1, 2), dtype=bool),
            )


def _triangle_key_points(
    offset: float = 0.0,
    confidence: float | None = None,
    visible: npt.NDArray[np.bool_] | None = None,
) -> KeyPoints:
    """Build one 3-point skeleton spanning a 50x70 box, shifted by `offset`."""
    xy = np.array([[[10, 10], [60, 10], [35, 80]]], dtype=np.float32) + offset
    detection_confidence = None if confidence is None else np.array([confidence])
    return KeyPoints(
        xy=xy,
        class_id=np.array([0]),
        detection_confidence=detection_confidence,
        visible=visible,
    )


TRIANGLE_SIGMAS = [0.25, 0.25, 0.25]


class TestKeyPointMeanAveragePrecision:
    """COCO keypoint mAP over `sv.KeyPoints` predictions and targets."""

    def test_exact_prediction_scores_one(self) -> None:
        """A prediction on top of its target scores 1 at every OKS threshold."""
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(
            _triangle_key_points(confidence=0.9), _triangle_key_points()
        ).compute()

        assert result.mAP_scores == pytest.approx(np.ones(10))

    def test_empty_predictions_score_zero(self) -> None:
        """Targets without any prediction give AP 0."""
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(KeyPoints.empty(), _triangle_key_points()).compute()

        assert result.map50_95 == pytest.approx(0.0)

    def test_empty_targets_give_sentinel(self) -> None:
        """Without targets there is nothing to score, so mAP is -1."""
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(_triangle_key_points(confidence=0.9), KeyPoints.empty())

        assert result.compute().map50_95 == -1

    @pytest.mark.parametrize(
        "target_data",
        [
            pytest.param({}, id="fallback-area"),
            pytest.param({"area": np.array([2000.0])}, id="data-area"),
        ],
    )
    def test_three_column_xy_matches_planar_xy(
        self, target_data: dict[str, npt.NDArray[np.float64]]
    ) -> None:
        """A z column in `xy` is ignored, so scores match the `(N, K, 2)` input."""

        def with_z(key_points: KeyPoints) -> KeyPoints:
            z = np.full((*key_points.xy.shape[:2], 1), 7.0, dtype=np.float32)
            return KeyPoints(
                xy=np.concatenate([key_points.xy, z], axis=2),
                class_id=key_points.class_id,
                detection_confidence=key_points.detection_confidence,
                visible=key_points.visible,
                data=key_points.data,
            )

        target = _triangle_key_points()
        target.data = dict(target_data)
        prediction = _triangle_key_points(offset=2.0, confidence=0.9)

        planar = (
            KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)
            .update(prediction, target)
            .compute()
        )
        spatial = (
            KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)
            .update(with_z(prediction), with_z(target))
            .compute()
        )

        assert spatial.mAP_scores == pytest.approx(planar.mAP_scores)

    def test_target_without_visible_points_is_skipped(self) -> None:
        """A target with no labelled keypoint is not counted as a miss."""
        hidden_target = _triangle_key_points(
            offset=300.0, visible=np.zeros((1, 3), dtype=bool)
        )
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(
            [_triangle_key_points(confidence=0.9), KeyPoints.empty()],
            [_triangle_key_points(), hidden_target],
        ).compute()

        assert result.map50_95 == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "area_data",
        [
            pytest.param({}, id="box-area"),
            pytest.param({"area": np.array([100.0])}, id="small-area"),
            pytest.param({"area": np.array([5000.0])}, id="medium-area"),
            pytest.param({"area": np.array([20000.0])}, id="large-area"),
        ],
    )
    def test_target_without_visible_points_with_box_is_ignore_region(
        self, area_data: dict[str, npt.NDArray[np.float64]]
    ) -> None:
        """With a box, a prediction on a hidden target is ignored, not a false positive.

        As in `pycocotools`, the hidden target is an ignore region in every size
        bucket, whatever its area, so it is neither a miss nor a match.
        """
        hidden_target = _triangle_key_points(
            offset=300.0, visible=np.zeros((1, 3), dtype=bool)
        )
        hidden_target.data["xyxy"] = np.array([[310.0, 310.0, 360.0, 380.0]])
        hidden_target.data.update(area_data)
        prediction_on_hidden = _triangle_key_points(offset=300.0, confidence=0.95)
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(
            [_triangle_key_points(confidence=0.9), prediction_on_hidden],
            [_triangle_key_points(), hidden_target],
        ).compute()

        assert result.map50_95 == pytest.approx(1.0)
        assert result.medium_objects is not None
        assert result.medium_objects.map50_95 == pytest.approx(1.0)

    def test_prediction_on_hidden_target_without_box_is_false_positive(
        self,
    ) -> None:
        """Without a box, a hidden target is skipped and its prediction is unmatched."""
        hidden_target = _triangle_key_points(
            offset=300.0, visible=np.zeros((1, 3), dtype=bool)
        )
        prediction_on_hidden = _triangle_key_points(offset=300.0, confidence=0.95)
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(
            [_triangle_key_points(confidence=0.9), prediction_on_hidden],
            [_triangle_key_points(), hidden_target],
        ).compute()

        assert result.map50 == pytest.approx(0.5)

    @pytest.mark.parametrize("side", ["predictions", "targets"])
    def test_raises_for_invalid_box_shape(self, side: str) -> None:
        """`data["xyxy"]` must hold one box per skeleton on either side."""
        prediction = _triangle_key_points(confidence=0.9)
        target = _triangle_key_points()
        inputs = {"predictions": prediction, "targets": target}
        inputs[side].data["xyxy"] = np.zeros((1, 2))
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        with pytest.raises(ValueError, match=f"{side}.data"):
            metric.update(prediction, target)

    @pytest.mark.parametrize(
        ("false_positive_data", "expected_medium_map50"),
        [
            pytest.param({}, 0.5, id="keypoint-span-medium"),
            pytest.param(
                {"xyxy": np.array([[500.0, 500.0, 700.0, 700.0]])},
                1.0,
                id="box-large",
            ),
        ],
    )
    def test_prediction_box_sets_object_size_bucket(
        self,
        false_positive_data: dict[str, npt.NDArray[np.float64]],
        expected_medium_map50: float,
    ) -> None:
        """An unmatched prediction is bucketed by its `data["xyxy"]` box area.

        The false positive sits in its own image, next to one with a match.

        Its keypoints span a medium box, so without a box it is a medium false
        positive. A large box moves it out of the medium bucket, as the
        `bbox` of a COCO result entry does in `pycocotools`.
        """
        target = _triangle_key_points()
        target.data["area"] = np.array([5000.0])
        false_positive = _triangle_key_points(offset=500.0, confidence=0.95)
        false_positive.data.update(false_positive_data)
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(
            [false_positive, _triangle_key_points(confidence=0.9)],
            [KeyPoints.empty(), target],
        ).compute()

        assert result.medium_objects is not None
        assert result.medium_objects.map50 == pytest.approx(expected_medium_map50)

    @pytest.mark.parametrize(
        "visible",
        [
            pytest.param(None, id="visible-keypoints"),
            pytest.param(np.zeros((1, 3), dtype=bool), id="hidden-keypoints"),
        ],
    )
    def test_crowd_target_absorbs_any_number_of_predictions(
        self, visible: npt.NDArray[np.bool_] | None
    ) -> None:
        """Predictions matching a crowd target are ignored, however many there are.

        A crowd target is never a miss, and unlike a regular ignore target it can match
        several predictions, with or without visible keypoints.
        """
        crowd = _triangle_key_points(offset=300.0, visible=visible)
        crowd.data["iscrowd"] = np.array([True])
        crowd.data["xyxy"] = np.array([[310.0, 310.0, 360.0, 380.0]])
        on_crowd = KeyPoints(
            xy=np.repeat(crowd.xy, 3, axis=0),
            class_id=np.zeros(3, dtype=int),
            detection_confidence=np.array([0.97, 0.96, 0.95]),
        )
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(
            [_triangle_key_points(confidence=0.9), on_crowd],
            [_triangle_key_points(), crowd],
        ).compute()

        assert result.map50_95 == pytest.approx(1.0)

    def test_raises_for_invalid_crowd_shape(self) -> None:
        """`targets.data["iscrowd"]` must hold one flag per target."""
        target = _triangle_key_points()
        target.data["iscrowd"] = np.zeros((1, 2))
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        with pytest.raises(ValueError, match="iscrowd"):
            metric.update(_triangle_key_points(confidence=0.9), target)

    def test_invisible_target_points_do_not_affect_oks(self) -> None:
        """A prediction far off on an unlabelled target point still matches."""
        prediction = _triangle_key_points(confidence=0.9)
        prediction.xy[0, 2] = [500, 500]
        target = _triangle_key_points(visible=np.array([[True, True, False]]))
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(prediction, target).compute()

        assert result.map50_95 == pytest.approx(1.0)

    def test_target_area_from_data_scales_oks(self) -> None:
        """A larger `data["area"]` tolerates a larger keypoint offset."""
        prediction = _triangle_key_points(offset=4.0, confidence=0.9)
        target = _triangle_key_points()
        target.data["area"] = np.array([100_000.0])
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(prediction, target).compute()

        assert result.map50_95 == pytest.approx(1.0)

    def test_raises_without_sigmas_for_non_coco_skeleton(self) -> None:
        """A 3-point skeleton without sigmas is rejected on update."""
        metric = KeyPointMeanAveragePrecision()

        with pytest.raises(ValueError, match="sigmas"):
            metric.update(_triangle_key_points(confidence=0.9), _triangle_key_points())

    def test_raises_for_mixed_skeletons(self) -> None:
        """Every skeleton evaluated together must have the same keypoint count."""
        other_skeleton = KeyPoints(xy=np.zeros((1, 4, 2), dtype=np.float32))
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        with pytest.raises(ValueError, match="same number of keypoints"):
            metric.update(_triangle_key_points(confidence=0.9), other_skeleton)

    def test_raises_for_mismatched_image_counts(self) -> None:
        """Predictions and targets must be given for the same images."""
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        with pytest.raises(ValueError, match="must be the same"):
            metric.update([KeyPoints.empty()], [KeyPoints.empty()] * 2)


class TestKeyPointMeanAveragePrecisionTargetArea:
    """OKS is normalized by a target area that falls back to a box when absent."""

    def test_fallback_area_spans_only_visible_keypoints(self) -> None:
        """Without `data["area"]`, the area is that of the visible keypoints' box.

        With the first and last triangle points visible the box is 25 x 70 = 1750, so a
        (10, 10) offset gives OKS = exp(-4 * 10**2 / 1750) = 0.796, which passes the OKS
        thresholds up to 0.75. Spanning all three points (area 3500) would give OKS
        0.892 and pass up to 0.85, so the unlabelled point must not count.
        """
        target = _triangle_key_points(visible=np.array([[True, False, True]]))
        prediction = _triangle_key_points(offset=10.0, confidence=0.9)
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(prediction, target).compute()

        assert result.mAP_scores == pytest.approx([1.0] * 6 + [0.0] * 4)

    @pytest.mark.parametrize(
        ("shift", "expected_map"),
        [
            pytest.param(0.0, 1.0, id="exact-keypoint-matches"),
            pytest.param(1.0, 0.0, id="one-pixel-off-keypoint-misses"),
        ],
    )
    def test_single_visible_keypoint_has_zero_area(
        self, shift: float, expected_map: float
    ) -> None:
        """A single visible keypoint gives area 0, so only an exact hit adds to OKS.

        The prediction is far off on the two unlabelled keypoints, which must not matter
        either way.
        """
        target = _triangle_key_points(visible=np.array([[True, False, False]]))
        prediction = _triangle_key_points(confidence=0.9)
        prediction.xy[0, 1:] += 200.0
        prediction.xy[0, 0] += shift
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(prediction, target).compute()

        assert result.map50_95 == pytest.approx(expected_map)

    def test_hidden_target_area_falls_back_to_box_area(self) -> None:
        """A hidden target without `data["area"]` is normalized by its box area.

        The box `(300, 300, 400, 380)` has area 8000 and expands to `[200, 500] x [220,
        460]`. A prediction with all keypoints 40 px right of the expansion has OKS =
        exp(-40**2 / (0.5 * 8000)) = 0.670: it is ignored at OKS thresholds up to 0.65
        and a false positive ranked above the match after that.
        """
        hidden_target = _triangle_key_points(
            offset=300.0, visible=np.zeros((1, 3), dtype=bool)
        )
        hidden_target.data["xyxy"] = np.array([[300.0, 300.0, 400.0, 380.0]])
        prediction_near_hidden = KeyPoints(
            xy=np.full((1, 3, 2), [540.0, 300.0], dtype=np.float32),
            class_id=np.array([0]),
            detection_confidence=np.array([0.95]),
        )
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(
            [_triangle_key_points(confidence=0.9), prediction_near_hidden],
            [_triangle_key_points(), hidden_target],
        ).compute()

        assert result.mAP_scores == pytest.approx([1.0] * 4 + [0.5] * 6)


class TestKeyPointMeanAveragePrecisionIgnoreRegions:
    """Ignore regions and crowd targets absorb predictions differently."""

    @pytest.mark.parametrize(
        ("is_crowd", "expected_map"),
        [
            pytest.param(False, 0.5, id="regular-second-is-false-positive"),
            pytest.param(True, 1.0, id="crowd-absorbs-both"),
        ],
    )
    def test_second_prediction_on_ignore_region(
        self, is_crowd: bool, expected_map: float
    ) -> None:
        """Only a crowd target absorbs a second prediction; a regular one does not.

        Two predictions (0.96, 0.95) sit on a hidden target with a box, next to an image
        holding one match (0.9). The first prediction is ignored by either kind of
        target. The second is ignored by a crowd target but unmatched, so a false
        positive ranked above the match, for a regular one: precision 1/2 at full recall
        gives AP 0.5.
        """
        hidden_target = _triangle_key_points(
            offset=300.0, visible=np.zeros((1, 3), dtype=bool)
        )
        hidden_target.data["xyxy"] = np.array([[310.0, 310.0, 360.0, 380.0]])
        hidden_target.data["iscrowd"] = np.array([is_crowd])
        two_on_hidden = KeyPoints(
            xy=np.repeat(hidden_target.xy, 2, axis=0),
            class_id=np.zeros(2, dtype=int),
            detection_confidence=np.array([0.96, 0.95]),
        )
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(
            [_triangle_key_points(confidence=0.9), two_on_hidden],
            [_triangle_key_points(), hidden_target],
        ).compute()

        assert result.mAP_scores == pytest.approx([expected_map] * 10)

    def test_crowd_and_regular_target_compete_for_one_prediction(self) -> None:
        """A prediction fitting a crowd and a regular target goes to the regular one.

        The crowd target is listed first and fits equally well, yet matching it would
        ignore the prediction and leave the regular target a miss (AP 0).
        """
        xy = np.repeat(_triangle_key_points().xy, 2, axis=0)
        targets = KeyPoints(
            xy=xy,
            class_id=np.zeros(2, dtype=int),
            data={"iscrowd": np.array([True, False])},
        )
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(_triangle_key_points(confidence=0.9), targets).compute()

        assert result.mAP_scores == pytest.approx(np.ones(10))

    def test_crowd_only_image_gives_sentinel(self) -> None:
        """With only a crowd target there is nothing to score, so mAP is -1."""
        crowd = _triangle_key_points()
        crowd.data["iscrowd"] = np.array([True])
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(_triangle_key_points(confidence=0.9), crowd).compute()

        assert result.map50_95 == -1
        assert result.map50 == -1


def _false_positives(
    count: int, class_id: int = 0, confidence: float = 0.9
) -> KeyPoints:
    """Build `count` predictions of one class, all far from every triangle target."""
    return KeyPoints(
        xy=np.repeat(_triangle_key_points(offset=500.0).xy, count, axis=0),
        class_id=np.full(count, class_id),
        detection_confidence=np.full(count, confidence),
    )


class TestKeyPointMeanAveragePrecisionMaxDetections:
    """At most 20 predictions per image and class are scored, highest score first."""

    @pytest.mark.parametrize(
        ("num_false_positives", "expected_map"),
        [
            pytest.param(19, 1 / 20, id="match-at-rank-20-counts"),
            pytest.param(20, 0.0, id="match-at-rank-21-is-dropped"),
        ],
    )
    def test_prediction_ranked_past_the_cap_is_dropped(
        self, num_false_positives: int, expected_map: float
    ) -> None:
        """A match counts at rank 20 and is lost at rank 21.

        Higher-scored false positives precede a lone match (0.5). At rank 20 the
        precision at full recall is 1/20, so AP is 0.05. At rank 21 the match is cut
        before it can be counted, so the target is a miss and AP is 0.
        """
        false_positives = _false_positives(num_false_positives)
        match = _triangle_key_points(confidence=0.5)
        predictions = KeyPoints(
            xy=np.concatenate([false_positives.xy, match.xy]),
            class_id=np.zeros(num_false_positives + 1, dtype=int),
            detection_confidence=np.append(false_positives.detection_confidence, 0.5),
        )
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(predictions, _triangle_key_points()).compute()

        assert result.map50_95 == pytest.approx(expected_map)

    def test_cap_applies_per_class(self) -> None:
        """Twenty higher-scored predictions of another class do not push a match out.

        Class 1 holds 20 false positives (0.9) and class 0 one match (0.5) in the same
        image. Counted per image, the match would be the 21st prediction and be cut.
        """
        match = _triangle_key_points(confidence=0.5)
        false_positives = _false_positives(20, class_id=1)
        predictions = KeyPoints(
            xy=np.concatenate([false_positives.xy, match.xy]),
            class_id=np.append(false_positives.class_id, 0),
            detection_confidence=np.append(false_positives.detection_confidence, 0.5),
        )
        class_one_target = _triangle_key_points(offset=1000.0)
        class_one_target.class_id = np.array([1])
        targets = KeyPoints(
            xy=np.concatenate([_triangle_key_points().xy, class_one_target.xy]),
            class_id=np.array([0, 1]),
        )
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(predictions, targets).compute()

        assert list(result.matched_classes) == [0, 1]
        assert result.ap_per_class[0] == pytest.approx(np.ones(10))

    def test_cap_applies_per_image(self) -> None:
        """Twenty higher-scored predictions in another image do not push a match out.

        Image 0 has 20 false positives (0.9) and no target, image 1 has one match (0.5).
        The match is within its own image's cap but ranks 21st overall, so precision at
        full recall is 1/21. A cap shared by all images would cut it.
        """
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(
            [_false_positives(20), _triangle_key_points(confidence=0.5)],
            [KeyPoints.empty(), _triangle_key_points()],
        ).compute()

        assert result.map50_95 == pytest.approx(1 / 21)

    @pytest.mark.parametrize("match_confidence", [None, 0.0])
    def test_missing_detection_confidence_scores_zero(
        self, match_confidence: float | None
    ) -> None:
        """Predictions without `detection_confidence` rank like score 0.

        The match scores 0, below a false positive (0.1) in another image, so precision
        at full recall is 1/2 and AP is 0.5. An explicit 0.0 agrees.
        """
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(
            [
                _triangle_key_points(confidence=match_confidence),
                _false_positives(1, confidence=0.1),
            ],
            [_triangle_key_points(), KeyPoints.empty()],
        ).compute()

        assert result.map50_95 == pytest.approx(0.5)


class TestKeyPointMeanAveragePrecisionParity:
    """Scores match COCO keypoint evaluation on synthetic data."""

    def test_scores_match_pycocotools(self) -> None:
        """Overall, per-size and per-class AP equal pycocotools 2.0.11."""
        images = _make_synthetic_pose_images()
        metric = KeyPointMeanAveragePrecision()

        result = metric.update(
            [image.predictions for image in images],
            [image.targets for image in images],
        ).compute()

        # pycocotools 2.0.11 `COCOeval(..., "keypoints")` on the same data, with
        # `area` from `targets.data["area"]`, v=2 for visible target points and
        # v=0 otherwise: stats[0:5] and the per-category mean of
        # `eval["precision"]` over recall thresholds (area all, maxDets 20).
        assert result.map50_95 == pytest.approx(EXPECTED_STATS[0], abs=PARITY_TOLERANCE)
        assert result.map50 == pytest.approx(EXPECTED_STATS[1], abs=PARITY_TOLERANCE)
        assert result.map75 == pytest.approx(EXPECTED_STATS[2], abs=PARITY_TOLERANCE)
        assert result.medium_objects is not None
        assert result.large_objects is not None
        assert result.medium_objects.map50_95 == pytest.approx(
            EXPECTED_STATS[3], abs=PARITY_TOLERANCE
        )
        assert result.large_objects.map50_95 == pytest.approx(
            EXPECTED_STATS[4], abs=PARITY_TOLERANCE
        )
        assert result.ap_per_class == pytest.approx(
            EXPECTED_AP_PER_CLASS, abs=PARITY_TOLERANCE
        )

    @pytest.mark.parametrize("seed", [7, 17, 42])
    def test_scores_match_faster_coco_eval(self, seed: int) -> None:
        """Overall and per-class scores match Faster COCO Eval on synthetic batches."""
        images = _make_synthetic_pose_images(seed)
        dataset, predictions = _to_coco(images)
        coco_targets = COCO()
        coco_targets.dataset = dataset
        coco_targets.createIndex()
        coco_predictions = coco_targets.loadRes(predictions)
        evaluator = COCOeval_faster(
            coco_targets,
            coco_predictions,
            iouType="keypoints",
            print_function=lambda *_: None,
        )
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        result = (
            KeyPointMeanAveragePrecision()
            .update(
                [image.predictions for image in images],
                [image.targets for image in images],
            )
            .compute()
        )

        expected_stats = evaluator.stats
        actual_stats = np.array(
            [
                result.map50_95,
                result.map50,
                result.map75,
                result.medium_objects.map50_95,
                result.large_objects.map50_95,
            ]
        )
        np.testing.assert_allclose(
            actual_stats, expected_stats[:5], atol=PARITY_TOLERANCE, rtol=0
        )
        expected_ap_per_class = (
            evaluator.eval["precision"][:, :, :, 0, -1].mean(axis=1).T
        )
        np.testing.assert_allclose(
            result.ap_per_class,
            expected_ap_per_class,
            atol=PARITY_TOLERANCE,
            rtol=0,
        )

    def test_evaluates_only_pycocotools_keypoint_area_ranges(self) -> None:
        """The evaluator keeps the all, medium and large ranges of `setKpParams`."""
        # Arrange
        dataset = EvaluationDataset(
            targets={"images": [], "annotations": [], "categories": []}
        )

        # Act
        evaluator = _KeyPointCOCOEvaluator(
            dataset, dataset, sigmas=np.ones(17), target_boxes={}
        )

        # Assert
        # pycocotools `Params.setKpParams` sets `areaRng` to these ranges; the
        # small bucket of box evaluation is not computed.
        assert evaluator.params.area_range == [
            [0**2, 1e5**2],
            [32**2, 96**2],
            [96**2, 1e5**2],
        ]


def _exact_match_result() -> KeyPointMeanAveragePrecisionResult:
    """Compute the result for one prediction on top of its target."""
    metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)
    return metric.update(
        _triangle_key_points(confidence=0.9), _triangle_key_points()
    ).compute()


class TestKeyPointMeanAveragePrecisionCategories:
    """Skeletons are grouped by class unless the metric is class agnostic."""

    def test_class_agnostic_matches_across_classes(self) -> None:
        """A class-agnostic metric matches a prediction of another class."""
        prediction = _triangle_key_points(confidence=0.9)
        prediction.class_id = np.array([1])
        metric = KeyPointMeanAveragePrecision(
            sigmas=TRIANGLE_SIGMAS, class_agnostic=True
        )

        result = metric.update(prediction, _triangle_key_points()).compute()

        assert result.map50_95 == pytest.approx(1.0)
        assert result.is_class_agnostic

    def test_missing_class_id_falls_back_to_class_zero(self) -> None:
        """Skeletons without `class_id` are scored as class 0."""
        prediction = _triangle_key_points(confidence=0.9)
        target = _triangle_key_points()
        prediction.class_id = None
        target.class_id = None
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(prediction, target).compute()

        assert result.map50_95 == pytest.approx(1.0)
        assert list(result.matched_classes) == [0]

    @pytest.mark.parametrize(
        ("prediction_class_id", "expected_classes"),
        [
            pytest.param(None, [0], id="no-class-ids-keep-class-zero"),
            pytest.param(np.array([1]), [-1], id="any-class-id-joins-minus-one"),
        ],
    )
    def test_class_agnostic_category_follows_mean_average_precision(
        self,
        prediction_class_id: npt.NDArray[np.int_] | None,
        expected_classes: list[int],
    ) -> None:
        """Class-agnostic skeletons share class -1, or keep 0 when no input has IDs.

        This mirrors `MeanAveragePrecision`, and an unlabelled target still matches
        a labelled prediction in the mixed case.
        """
        prediction = _triangle_key_points(confidence=0.9)
        prediction.class_id = prediction_class_id
        target = _triangle_key_points()
        target.class_id = None
        metric = KeyPointMeanAveragePrecision(
            sigmas=TRIANGLE_SIGMAS, class_agnostic=True
        )

        result = metric.update(prediction, target).compute()

        assert list(result.matched_classes) == expected_classes
        assert result.map50_95 == pytest.approx(1.0)


class TestKeyPointMeanAveragePrecisionReset:
    """`reset` discards everything stored by earlier updates."""

    def test_reset_clears_stored_data(self) -> None:
        """After `reset`, earlier updates no longer contribute to the score."""
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)
        metric.update(_triangle_key_points(confidence=0.9), _triangle_key_points())

        metric.reset()
        result = metric.update(KeyPoints.empty(), _triangle_key_points()).compute()

        assert result.map50_95 == pytest.approx(0.0)


class TestKeyPointMeanAveragePrecisionResult:
    """Summary, DataFrame and plot helpers of the keypoint mAP result."""

    def test_str_lists_overall_and_size_scores(self) -> None:
        """The summary prints three overall OKS lines, then medium and large.

        An exact match scores 1 overall and for the medium bucket its keypoints span,
        and -1 for the large bucket, which holds no target. This is the example in
        the `__str__` docstring.
        """
        result = _exact_match_result()

        lines = str(result).splitlines()

        prefix = "Average Precision (AP) @[ OKS="
        suffix = "maxDets= 20 ] = "
        assert lines == [
            f"{prefix}0.50:0.95 | area=   all | {suffix}1.000",
            f"{prefix}0.50      | area=   all | {suffix}1.000",
            f"{prefix}0.75      | area=   all | {suffix}1.000",
            f"{prefix}0.50:0.95 | area=medium | {suffix}1.000",
            f"{prefix}0.50:0.95 | area= large | {suffix}-1.000",
        ]

    def test_to_pandas_holds_overall_scores(self) -> None:
        """The DataFrame has one row with the overall mAP columns."""
        result = _exact_match_result()

        data_frame = result.to_pandas()

        assert len(data_frame) == 1
        assert data_frame["mAP@50:95"].iloc[0] == pytest.approx(1.0)
        assert data_frame["mAP@50"].iloc[0] == pytest.approx(1.0)
        assert data_frame["mAP@75"].iloc[0] == pytest.approx(1.0)

    @pytest.mark.parametrize("include_object_sizes", [True, False])
    def test_plot_details_start_with_overall_scores(
        self, include_object_sizes: bool
    ) -> None:
        """Plot details list the overall scores, then any object-size bars."""
        result = _exact_match_result()

        details = result._get_plot_details(include_object_sizes=include_object_sizes)

        assert details.labels[:3] == ["mAP@50:95", "mAP@50", "mAP@75"]
        assert details.values[:3] == pytest.approx([1.0, 1.0, 1.0])
        assert len(details.labels) == len(details.values) == len(details.colors)
        assert (len(details.labels) > 3) == include_object_sizes
        assert "Keypoint Mean Average Precision" in details.title

    def test_plot_shows_figure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`plot` draws the bars and hands them to `plt.show` once."""
        from matplotlib import pyplot as plt

        shown: list[bool] = []
        monkeypatch.setattr(plt, "show", lambda: shown.append(True))
        result = _exact_match_result()

        result.plot()
        plt.close("all")

        assert shown == [True]


class TestKeyPointMeanAveragePrecisionInputValidation:
    """Invalid inputs are rejected up front and leave the metric unchanged."""

    @pytest.mark.parametrize(
        "sigmas",
        [
            pytest.param([0.1, 0.0, 0.1], id="zero"),
            pytest.param([0.1, -0.1, 0.1], id="negative"),
            pytest.param([0.1, np.nan, 0.1], id="nan"),
            pytest.param([0.1, np.inf, 0.1], id="infinite"),
        ],
    )
    def test_constructor_raises_for_invalid_sigmas(self, sigmas: list[float]) -> None:
        """Sigmas that are not positive and finite are rejected before any update."""
        with pytest.raises(ValueError, match="positive and finite"):
            KeyPointMeanAveragePrecision(sigmas=sigmas)

    @pytest.mark.parametrize(
        ("area", "match"),
        [
            pytest.param([[2000.0]], "must have shape", id="column-vector"),
            pytest.param([2000.0, 2000.0], "must have shape", id="one-too-many"),
            pytest.param([np.nan], "must be finite and non-negative", id="nan"),
            pytest.param([np.inf], "must be finite and non-negative", id="infinite"),
            pytest.param([-5.0], "must be finite and non-negative", id="negative"),
        ],
    )
    def test_update_raises_for_invalid_target_area(
        self, area: list[float] | list[list[float]], match: str
    ) -> None:
        """`targets.data["area"]` must hold one finite, non-negative area per target."""
        target = _triangle_key_points()
        target.data["area"] = np.array(area)
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        with pytest.raises(ValueError, match=rf"targets\.data\['area'\]` {match}"):
            metric.update(_triangle_key_points(confidence=0.9), target)

    def test_update_accepts_zero_target_area(self) -> None:
        """A zero area is accepted, and an exact match then still scores 1."""
        target = _triangle_key_points()
        target.data["area"] = np.array([0.0])
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(_triangle_key_points(confidence=0.9), target).compute()

        assert result.map50_95 == pytest.approx(1.0)

    def test_rejected_update_does_not_pin_skeleton_size(self) -> None:
        """A rejected update records no skeleton size for later updates.

        The 4-point skeleton must then fail on its own sigma length, not with the
        misleading "earlier skeletons" error that a leaked 3-point size gave.
        """
        bad_target = _triangle_key_points()
        bad_target.data["iscrowd"] = np.zeros((1, 2))
        four_points = KeyPoints(xy=np.zeros((1, 4, 2), dtype=np.float32))
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)
        with pytest.raises(ValueError, match="iscrowd"):
            metric.update(_triangle_key_points(confidence=0.9), bad_target)

        with pytest.raises(ValueError, match=r"`sigmas` must have shape \(4,\)"):
            metric.update(four_points, four_points)

    @pytest.mark.parametrize(
        "sigmas",
        [
            pytest.param([], id="explicit-empty-sigmas"),
            pytest.param(None, id="default-sigmas"),
        ],
    )
    def test_update_raises_for_skeleton_without_keypoints(
        self, sigmas: list[float] | None
    ) -> None:
        """A skeleton with zero keypoints is rejected on update, not deep in compute.

        Empty sigmas pass the constructor check, so `update` is where it must fail.
        """
        no_keypoints = KeyPoints(xy=np.zeros((1, 0, 2), dtype=np.float32))
        metric = KeyPointMeanAveragePrecision(sigmas=sigmas)

        with pytest.raises(ValueError, match="at least one keypoint"):
            metric.update(no_keypoints, no_keypoints)


class TestKeyPointMeanAveragePrecisionNonFiniteKeypoints:
    """Non-finite target keypoints are unlabelled; non-finite predictions add 0."""

    @pytest.mark.parametrize(
        ("value", "data"),
        [
            pytest.param(np.nan, {}, id="nan-without-area"),
            pytest.param(np.inf, {}, id="inf-without-area"),
            pytest.param(np.nan, {"area": np.array([2000.0])}, id="nan-with-area"),
            pytest.param(np.inf, {"area": np.array([2000.0])}, id="inf-with-area"),
        ],
    )
    def test_non_finite_target_keypoint_is_unlabelled(
        self, value: float, data: dict[str, npt.NDArray[np.float64]]
    ) -> None:
        """A non-finite target keypoint is left out, like one with `visible=False`.

        The remaining keypoints then decide the OKS and, without `data["area"]`, the
        fallback area, so an exact prediction scores 1 at every threshold.
        """
        target = _triangle_key_points()
        target.xy[0, 1, 0] = value
        target.data = data
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(_triangle_key_points(confidence=0.9), target).compute()

        assert result.map50_95 == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("data", "expected_map50"),
        [
            pytest.param(
                {"area": np.array([2000.0])}, 0.5, id="without-box-is-skipped"
            ),
            pytest.param(
                {
                    "area": np.array([2000.0]),
                    "xyxy": np.array([[310.0, 310.0, 360.0, 380.0]]),
                },
                1.0,
                id="with-box-is-ignore-region",
            ),
        ],
    )
    def test_target_with_only_non_finite_keypoints_is_hidden(
        self, data: dict[str, npt.NDArray[np.float64]], expected_map50: float
    ) -> None:
        """A target with only non-finite keypoints counts as one without visible ones.

        With a box it is an ignore region, so a prediction on it is ignored. Without one
        it is skipped, so that prediction counts as a false positive.
        """
        hidden_target = KeyPoints(
            xy=np.full((1, 3, 2), np.nan, dtype=np.float32),
            class_id=np.array([0]),
            data=data,
        )
        prediction_on_hidden = _triangle_key_points(offset=300.0, confidence=0.95)
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(
            [_triangle_key_points(confidence=0.9), prediction_on_hidden],
            [_triangle_key_points(), hidden_target],
        ).compute()

        assert result.map50 == pytest.approx(expected_map50)

    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_non_finite_prediction_keypoint_adds_zero(self, value: float) -> None:
        """A non-finite prediction keypoint adds 0 while its finite keypoints count.

        Two of three exact keypoints give OKS 2/3, which clears the thresholds up to
        0.65 and none above.
        """
        prediction = _triangle_key_points(confidence=0.9)
        prediction.xy[0, 1, 0] = value
        target = _triangle_key_points()
        target.data["area"] = np.array([2000.0])
        metric = KeyPointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(prediction, target).compute()

        assert result.mAP_scores == pytest.approx([1.0] * 4 + [0.0] * 6)
