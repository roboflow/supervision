from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

from supervision.detection.utils.iou_and_nms import (
    _COCO_KEYPOINT_SIGMAS,
    _keypoint_oks_batch,
)
from supervision.key_points.core import KeyPoints
from supervision.metrics.keypoint_mean_average_precision import (
    KeypointMeanAveragePrecision,
    KeypointMeanAveragePrecisionResult,
)

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


# pycocotools 2.0.11 results for `_make_synthetic_pose_images()`.
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


class TestKeypointOksBatch:
    """Pairwise OKS between target and detected keypoint sets."""

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

    @pytest.mark.parametrize(("num_true", "num_detection"), [(0, 2), (2, 0), (0, 0)])
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
        ],
    )
    def test_raises_for_invalid_sigmas(self, sigmas: list[float]) -> None:
        """Sigmas must hold one positive value per keypoint."""
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


class TestKeypointMeanAveragePrecision:
    """COCO keypoint mAP over `sv.KeyPoints` predictions and targets."""

    def test_exact_prediction_scores_one(self) -> None:
        """A prediction on top of its target scores 1 at every OKS threshold."""
        metric = KeypointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(
            _triangle_key_points(confidence=0.9), _triangle_key_points()
        ).compute()

        assert result.mAP_scores == pytest.approx(np.ones(10))

    def test_empty_predictions_score_zero(self) -> None:
        """Targets without any prediction give AP 0."""
        metric = KeypointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(KeyPoints.empty(), _triangle_key_points()).compute()

        assert result.map50_95 == pytest.approx(0.0)

    def test_empty_targets_give_sentinel(self) -> None:
        """Without targets there is nothing to score, so mAP is -1."""
        metric = KeypointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(_triangle_key_points(confidence=0.9), KeyPoints.empty())

        assert result.compute().map50_95 == -1

    def test_target_without_visible_points_is_skipped(self) -> None:
        """A target with no labelled keypoint is not counted as a miss."""
        hidden_target = _triangle_key_points(
            offset=300.0, visible=np.zeros((1, 3), dtype=bool)
        )
        metric = KeypointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(
            [_triangle_key_points(confidence=0.9), KeyPoints.empty()],
            [_triangle_key_points(), hidden_target],
        ).compute()

        assert result.map50_95 == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "area",
        [
            pytest.param(None, id="box-area"),
            pytest.param(100.0, id="small-area"),
            pytest.param(5000.0, id="medium-area"),
            pytest.param(20000.0, id="large-area"),
        ],
    )
    def test_target_without_visible_points_with_box_is_ignore_region(
        self, area: float | None
    ) -> None:
        """With a box, a prediction on a hidden target is ignored, not a false positive.

        As in `pycocotools`, the hidden target is an ignore region in every size
        bucket, whatever its area, so it is neither a miss nor a match.
        """
        hidden_target = _triangle_key_points(
            offset=300.0, visible=np.zeros((1, 3), dtype=bool)
        )
        hidden_target.data["xyxy"] = np.array([[310.0, 310.0, 360.0, 380.0]])
        if area is not None:
            hidden_target.data["area"] = np.array([area])
        prediction_on_hidden = _triangle_key_points(offset=300.0, confidence=0.95)
        metric = KeypointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

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
        metric = KeypointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(
            [_triangle_key_points(confidence=0.9), prediction_on_hidden],
            [_triangle_key_points(), hidden_target],
        ).compute()

        assert result.map50 == pytest.approx(0.5, abs=0.01)

    @pytest.mark.parametrize("side", ["predictions", "targets"])
    def test_raises_for_invalid_box_shape(self, side: str) -> None:
        """`data["xyxy"]` must hold one box per skeleton on either side."""
        prediction = _triangle_key_points(confidence=0.9)
        target = _triangle_key_points()
        bad = prediction if side == "predictions" else target
        bad.data["xyxy"] = np.zeros((1, 2))
        metric = KeypointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        with pytest.raises(ValueError, match=f"{side}.data"):
            metric.update(prediction, target)

    @pytest.mark.parametrize(
        ("false_positive_xyxy", "expected_medium_map50"),
        [
            pytest.param(None, 0.5, id="keypoint-span-medium"),
            pytest.param([[500.0, 500.0, 700.0, 700.0]], 1.0, id="box-large"),
        ],
    )
    def test_prediction_box_sets_object_size_bucket(
        self,
        false_positive_xyxy: list[list[float]] | None,
        expected_medium_map50: float,
    ) -> None:
        """An unmatched prediction is bucketed by its `data["xyxy"]` box area.

        The false positive sits in its own image, next to one with a match.

        Its keypoints span a medium box, so without a box it is a medium false
        positive. A large box moves it out of the medium bucket, as a result
        `bbox` does in `pycocotools`.
        """
        target = _triangle_key_points()
        target.data["area"] = np.array([5000.0])
        false_positive = _triangle_key_points(offset=500.0, confidence=0.95)
        if false_positive_xyxy is not None:
            false_positive.data["xyxy"] = np.array(false_positive_xyxy)
        metric = KeypointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(
            [false_positive, _triangle_key_points(confidence=0.9)],
            [KeyPoints.empty(), target],
        ).compute()

        assert result.medium_objects is not None
        assert result.medium_objects.map50 == pytest.approx(
            expected_medium_map50, abs=0.01
        )

    @pytest.mark.parametrize("hidden", [False, True])
    def test_crowd_target_absorbs_any_number_of_predictions(self, hidden: bool) -> None:
        """Predictions matching a crowd target are ignored, however many there are.

        A crowd target is never a miss, and unlike a regular ignore target it can match
        several predictions, with or without visible keypoints.
        """
        crowd = _triangle_key_points(offset=300.0)
        if hidden:
            crowd.visible = np.zeros((1, 3), dtype=bool)
        crowd.data["iscrowd"] = np.array([True])
        crowd.data["xyxy"] = np.array([[310.0, 310.0, 360.0, 380.0]])
        on_crowd = KeyPoints(
            xy=np.repeat(crowd.xy, 3, axis=0),
            class_id=np.zeros(3, dtype=int),
            detection_confidence=np.array([0.97, 0.96, 0.95]),
        )
        metric = KeypointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(
            [_triangle_key_points(confidence=0.9), on_crowd],
            [_triangle_key_points(), crowd],
        ).compute()

        assert result.map50_95 == pytest.approx(1.0)

    def test_raises_for_invalid_crowd_shape(self) -> None:
        """`targets.data["iscrowd"]` must hold one flag per target."""
        target = _triangle_key_points()
        target.data["iscrowd"] = np.zeros((1, 2))
        metric = KeypointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        with pytest.raises(ValueError, match="iscrowd"):
            metric.update(_triangle_key_points(confidence=0.9), target)

    def test_invisible_target_points_do_not_affect_oks(self) -> None:
        """A prediction far off on an unlabelled target point still matches."""
        prediction = _triangle_key_points(confidence=0.9)
        prediction.xy[0, 2] = [500, 500]
        target = _triangle_key_points(visible=np.array([[True, True, False]]))
        metric = KeypointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(prediction, target).compute()

        assert result.map50_95 == pytest.approx(1.0)

    def test_target_area_from_data_scales_oks(self) -> None:
        """A larger `data["area"]` tolerates a larger keypoint offset."""
        prediction = _triangle_key_points(offset=4.0, confidence=0.9)
        target = _triangle_key_points()
        target.data["area"] = np.array([100_000.0])
        metric = KeypointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(prediction, target).compute()

        assert result.map50_95 == pytest.approx(1.0)

    def test_raises_without_sigmas_for_non_coco_skeleton(self) -> None:
        """A 3-point skeleton without sigmas is rejected on update."""
        metric = KeypointMeanAveragePrecision()

        with pytest.raises(ValueError, match="sigmas"):
            metric.update(_triangle_key_points(confidence=0.9), _triangle_key_points())

    def test_raises_for_mixed_skeletons(self) -> None:
        """Every skeleton evaluated together must have the same keypoint count."""
        other_skeleton = KeyPoints(xy=np.zeros((1, 4, 2), dtype=np.float32))
        metric = KeypointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        with pytest.raises(ValueError, match="same number of keypoints"):
            metric.update(_triangle_key_points(confidence=0.9), other_skeleton)

    def test_raises_for_mismatched_image_counts(self) -> None:
        """Predictions and targets must be given for the same images."""
        metric = KeypointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        with pytest.raises(ValueError, match="must be the same"):
            metric.update([KeyPoints.empty()], [KeyPoints.empty()] * 2)


class TestKeypointMeanAveragePrecisionPycocotoolsParity:
    """Scores match `COCOeval(..., iouType="keypoints")` on synthetic data."""

    def test_scores_match_pycocotools(self) -> None:
        """Overall, per-size and per-class AP equal pycocotools 2.0.11."""
        images = _make_synthetic_pose_images()
        metric = KeypointMeanAveragePrecision()

        result = metric.update(
            [image.predictions for image in images],
            [image.targets for image in images],
        ).compute()

        # pycocotools 2.0.11 `COCOeval(..., "keypoints")` on the same data, with
        # `area` from `targets.data["area"]`, v=2 for visible target points and
        # v=0 otherwise: stats[0:5] and the per-category mean of
        # `eval["precision"]` over recall thresholds (area all, maxDets 20).
        assert result.map50_95 == pytest.approx(EXPECTED_STATS[0], abs=1e-6)
        assert result.map50 == pytest.approx(EXPECTED_STATS[1], abs=1e-6)
        assert result.map75 == pytest.approx(EXPECTED_STATS[2], abs=1e-6)
        assert result.medium_objects is not None
        assert result.large_objects is not None
        assert result.medium_objects.map50_95 == pytest.approx(
            EXPECTED_STATS[3], abs=1e-6
        )
        assert result.large_objects.map50_95 == pytest.approx(
            EXPECTED_STATS[4], abs=1e-6
        )
        assert result.ap_per_class == pytest.approx(EXPECTED_AP_PER_CLASS, abs=1e-6)


class TestKeypointOksBatchShapeValidation:
    """Per-target inputs of `_keypoint_oks_batch` must match the target count."""

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


def _exact_match_result() -> KeypointMeanAveragePrecisionResult:
    """Compute the result for one prediction on top of its target."""
    metric = KeypointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)
    return metric.update(
        _triangle_key_points(confidence=0.9), _triangle_key_points()
    ).compute()


class TestKeypointMeanAveragePrecisionCategories:
    """Skeletons are grouped by class unless the metric is class agnostic."""

    def test_class_agnostic_matches_across_classes(self) -> None:
        """A class-agnostic metric matches a prediction of another class."""
        prediction = _triangle_key_points(confidence=0.9)
        prediction.class_id = np.array([1])
        metric = KeypointMeanAveragePrecision(
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
        metric = KeypointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)

        result = metric.update(prediction, target).compute()

        assert result.map50_95 == pytest.approx(1.0)
        assert list(result.matched_classes) == [0]

    def test_reset_clears_stored_data(self) -> None:
        """After `reset`, earlier updates no longer contribute to the score."""
        metric = KeypointMeanAveragePrecision(sigmas=TRIANGLE_SIGMAS)
        metric.update(_triangle_key_points(confidence=0.9), _triangle_key_points())

        metric.reset()
        result = metric.update(KeyPoints.empty(), _triangle_key_points()).compute()

        assert result.map50_95 == pytest.approx(0.0)


class TestKeypointMeanAveragePrecisionResult:
    """Summary, DataFrame and plot helpers of the keypoint mAP result."""

    def test_str_lists_overall_scores(self) -> None:
        """The summary starts with the three overall OKS lines."""
        result = _exact_match_result()

        lines = str(result).splitlines()

        assert lines[0].startswith("Average Precision (AP) @[ OKS=0.50:0.95")
        assert lines[0].endswith("= 1.000")
        assert len(lines) == 3 + (result.medium_objects is not None) + (
            result.large_objects is not None
        )

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
