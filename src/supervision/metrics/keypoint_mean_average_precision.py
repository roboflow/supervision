from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from supervision.config import AREA_DATA_FIELD
from supervision.detection.utils.iou_and_nms import (
    _keypoint_oks_batch,
    _resolve_keypoint_sigmas,
)
from supervision.draw.color import LEGACY_COLOR_PALETTE
from supervision.key_points.core import KeyPoints
from supervision.metrics.core import (
    Metric,
    MetricResult,
    PlotDetails,
    _append_object_size_plot_details,
)
from supervision.metrics.mean_average_precision import (
    COCOEvaluator,
    EvaluationDataset,
    _mean_valid_score,
    _scores_to_pandas,
    _show_bar_plot,
    _TypeCocoDict,
)

if TYPE_CHECKING:
    import pandas as pd

_KEYPOINT_MAX_DETECTIONS = 20
"""Maximum detections per image and class in COCO keypoint evaluation."""

_XYXY_DATA_FIELD = "xyxy"
"""`KeyPoints.data` key of per-object `(N, 4)` boxes in `(x_min, y_min, x_max, y_max)`
format."""

_ISCROWD_DATA_FIELD = "iscrowd"
"""`targets.data` key of per-target COCO crowd flags of shape `(N,)`."""


@dataclass
class KeypointMeanAveragePrecisionResult(MetricResult):
    """The result of the keypoint Mean Average Precision calculation.

    Scores are `-1` when there are no targets to evaluate. COCO keypoint
    evaluation reports no small-object bucket, because small people are not
    annotated with keypoints in COCO, so only medium and large results exist.

    Attributes:
        is_class_agnostic: When computing class-agnostic results, class ID
            is set to `-1`.
        mAP_scores: the mAP scores at each OKS threshold.
            Shape: `(num_oks_thresholds,)`
        ap_per_class: the average precision scores per class and OKS threshold.
            Shape: `(num_target_classes, num_oks_thresholds)`
        oks_thresholds: the OKS thresholds used in the calculations.
        matched_classes: the class IDs of all target classes.
            Corresponds to the rows of `ap_per_class`.
        medium_objects: the mAP results for medium objects
            (32² ≤ area < 96²).
        large_objects: the mAP results for large objects (area ≥ 96²).
    """

    is_class_agnostic: bool

    @property
    def map50_95(self) -> float:
        """The mAP score at OKS thresholds from `0.5` to `0.95`."""
        return _mean_valid_score(self.mAP_scores)

    @property
    def map50(self) -> float:
        """The mAP score at OKS threshold of `0.5`."""
        return float(self.mAP_scores[0])

    @property
    def map75(self) -> float:
        """The mAP score at OKS threshold of `0.75`."""
        return float(self.mAP_scores[5])

    mAP_scores: npt.NDArray[np.float64]
    ap_per_class: npt.NDArray[np.float64]
    oks_thresholds: npt.NDArray[np.float64]
    matched_classes: npt.NDArray[np.int32]
    medium_objects: KeypointMeanAveragePrecisionResult | None = None
    large_objects: KeypointMeanAveragePrecisionResult | None = None

    def __str__(self) -> str:
        """Format the scores like the `pycocotools` keypoint summary.

        Example:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> from supervision.metrics import KeypointMeanAveragePrecision
            >>> xy = np.array([[[10, 10], [60, 10], [35, 80]]], dtype=np.float32)
            >>> targets = sv.KeyPoints(xy=xy, class_id=np.array([0]))
            >>> predictions = sv.KeyPoints(
            ...     xy=xy, class_id=np.array([0]), detection_confidence=np.array([0.9])
            ... )
            >>> metric = KeypointMeanAveragePrecision(sigmas=[0.5, 0.5, 0.5])
            >>> print(metric.update(predictions, targets).compute())
            Average Precision (AP) @[ OKS=0.50:0.95 | area=   all | ... ] = 1.000
            Average Precision (AP) @[ OKS=0.50      | area=   all | ... ] = 1.000
            Average Precision (AP) @[ OKS=0.75      | area=   all | ... ] = 1.000
            Average Precision (AP) @[ OKS=0.50:0.95 | area=medium | ... ] = 1.000
            Average Precision (AP) @[ OKS=0.50:0.95 | area= large | ... ] = -1.000

            ```
        """
        prefix = "Average Precision (AP) @[ OKS="
        suffix = f"maxDets={_KEYPOINT_MAX_DETECTIONS:>3d} ] = "
        lines = [
            f"{prefix}0.50:0.95 | area=   all | {suffix}{self.map50_95:.3f}",
            f"{prefix}0.50      | area=   all | {suffix}{self.map50:.3f}",
            f"{prefix}0.75      | area=   all | {suffix}{self.map75:.3f}",
        ]
        if self.medium_objects is not None:
            medium = self.medium_objects.map50_95
            lines.append(f"{prefix}0.50:0.95 | area=medium | {suffix}{medium:.3f}")
        if self.large_objects is not None:
            large = self.large_objects.map50_95
            lines.append(f"{prefix}0.50:0.95 | area= large | {suffix}{large:.3f}")
        return "\n".join(lines)

    def to_pandas(self) -> pd.DataFrame:
        """Convert the result to a pandas DataFrame.

        Returns:
            The result as a DataFrame.
        """
        return _scores_to_pandas(
            {"mAP@50:95": self.map50_95, "mAP@50": self.map50, "mAP@75": self.map75},
            [
                ("medium_objects", self.medium_objects),
                ("large_objects", self.large_objects),
            ],
        )

    def _get_plot_details(self, include_object_sizes: bool = True) -> PlotDetails:
        """Return bar-chart data for keypoint mAP scores.

        Args:
            include_object_sizes: When ``True``, include bars for the medium and
                large object-size categories.
        """
        metric_labels = ["mAP@50:95", "mAP@50", "mAP@75"]
        labels = list(metric_labels)
        values = [self.map50_95, self.map50, self.map75]
        colors = [LEGACY_COLOR_PALETTE[0]] * 3
        _append_object_size_plot_details(
            labels,
            values,
            colors,
            include_object_sizes=include_object_sizes,
            metric_labels=metric_labels,
            small_objects=None,
            medium_objects=self.medium_objects,
            large_objects=self.large_objects,
            value_getter=lambda result: [
                result.map50_95,
                result.map50,
                result.map75,
            ],
        )
        title = (
            "Keypoint Mean Average Precision\n"
            f"(class agnostic: {self.is_class_agnostic})"
        )
        return PlotDetails(labels=labels, values=values, colors=colors, title=title)

    def plot(self) -> None:
        """Plot the keypoint mAP results as a bar chart."""
        _show_bar_plot(self._get_plot_details())


class _KeypointCOCOEvaluator(COCOEvaluator):
    """COCO evaluator that matches by OKS instead of IoU.

    Matching, accumulation and the 101-point interpolation are inherited, so
    keypoint mAP shares them with box, mask and oriented-box mAP. Each
    annotation's `content` holds its keypoints as `(K, 3)` rows of
    `(x, y, visible)`. Boxes of targets without visible keypoints are kept in a
    separate mapping by annotation id, so the shared COCO dictionaries hold no
    keypoint-only fields.
    """

    def __init__(
        self,
        coco_targets: EvaluationDataset,
        coco_predictions: EvaluationDataset,
        sigmas: npt.NDArray[np.float64],
        target_boxes: dict[int, npt.NDArray[np.float64]],
    ) -> None:
        """Set up the evaluator with COCO keypoint max detections.

        Args:
            coco_targets: The dataset with the ground truths.
            coco_predictions: The dataset with the predictions.
            sigmas: Per-keypoint OKS sigmas.
            target_boxes: xyxy box by annotation id for targets without
                visible keypoints, which OKS measures against.
        """
        super().__init__(coco_targets, coco_predictions)
        self.params.max_dets = [_KEYPOINT_MAX_DETECTIONS]
        self._sigmas = sigmas
        self._target_boxes = target_boxes

    def _compute_iou(self, img_id: int, cat_id: int) -> npt.NDArray[np.float64]:
        """Compute OKS between the targets and the top-scored predictions.

        Args:
            img_id: The image id.
            cat_id: The category id.

        Returns:
            OKS matrix of shape `(predictions, targets)`.
        """
        gt = self._targets[img_id, cat_id]
        dt = self._predictions[img_id, cat_id]
        if len(gt) == 0 and len(dt) == 0:
            return np.array([], dtype=np.float64)

        inds = np.argsort([-d["score"] for d in dt], kind="stable")
        dt = [dt[i] for i in inds][: self.params.max_dets[-1]]
        if len(gt) == 0 or len(dt) == 0:
            return np.empty((len(dt), len(gt)), dtype=np.float64)

        gt_content = np.stack([g["content"] for g in gt])
        dt_content = np.stack([d["content"] for d in dt])
        xyxy_true = None
        if any(g["id"] in self._target_boxes for g in gt):
            # Rows of targets with visible keypoints are unused placeholders.
            xyxy_true = np.array(
                [self._target_boxes.get(g["id"], np.zeros(4)) for g in gt],
                dtype=np.float64,
            )
        oks = _keypoint_oks_batch(
            keypoints_true=gt_content[:, :, :2],
            keypoints_detection=dt_content[:, :, :2],
            area_true=np.array([g["area"] for g in gt], dtype=np.float64),
            sigmas=self._sigmas,
            visible_true=gt_content[:, :, 2] > 0,
            xyxy_true=xyxy_true,
        )
        return cast(npt.NDArray[np.float64], oks.T)


def _keypoints_bbox(xy: npt.NDArray[np.float64]) -> list[float]:
    """Return the COCO `[x, y, w, h]` box spanning the given `(K, 2)` keypoints.

    Args:
        xy: Keypoint coordinates of shape `(K, 2)`, with `K >= 1`.

    Returns:
        The box as `[x_min, y_min, width, height]`.
    """
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


class KeypointMeanAveragePrecision(Metric[KeypointMeanAveragePrecisionResult]):
    """Keypoint Mean Average Precision based on Object Keypoint Similarity (OKS).

    This is the COCO keypoint metric: predictions are matched to targets by OKS
    instead of IoU, at 10 OKS thresholds from `0.5` to `0.95`, with 101-point
    recall interpolation and at most 20 predictions per image and class. The
    matching and accumulation are shared with `MeanAveragePrecision`, and the
    result matches `pycocotools` `COCOeval(..., iouType="keypoints")` when the
    target areas match.

    Inputs are one `sv.KeyPoints` per image for predictions and targets:

    - **Visibility.** Target keypoints with `visible` set to `False` are
      unlabelled, like COCO `v=0`, and do not enter the OKS. `visible=None`
      means every keypoint is labelled. Prediction visibility is not used.
    - **Targets without visible keypoints** (COCO `num_keypoints == 0`). When
      `targets.data["xyxy"]` holds the target boxes as `(N, 4)`
      `(x_min, y_min, x_max, y_max)`, such a target is kept as an ignore
      region, as in `pycocotools`: OKS is measured to its box expanded by its
      own width and height on each side, a prediction matched to it is ignored
      rather than counted, and it is never a miss, in every size bucket. Its
      area is `targets.data["area"]` or, if absent, the box area. Without
      boxes the target is skipped, so a prediction that lands on it counts as
      a false positive. Boxes of targets with visible keypoints are not used.
    - **Area.** OKS is normalized by the target's object area. Pass it as
      `targets.data["area"]` (the same key `MeanAveragePrecision` reads) to
      reproduce COCO, which uses the annotated segmentation area. Without it,
      the area of the box spanning the target's visible keypoints is used. That
      box is usually tighter than the person's outline, so each pair's OKS
      differs from COCO; mAP can move either way, since matches change too. A
      target with a single visible keypoint gets area `0`. The fallback keeps
      the metric usable on keypoint-only labels.
    - **Scores.** Predictions are ranked by `detection_confidence`; keypoint
      `confidence` is not used. When `detection_confidence` is `None`, every
      prediction of that image scores `0`, as in `MeanAveragePrecision`.
    - **Object size.** Targets fall into the medium and large buckets by that
      same area. Predictions fall in by the area of their box in
      `predictions.data["xyxy"]` when given, else of the box spanning all
      their keypoints, as `pycocotools` does with and without a result
      `bbox`. Boxes never change the OKS of a prediction.
    - **Crowd.** `targets.data["iscrowd"]` marks crowd targets, as in COCO
      `iscrowd=1`. A crowd target is an ignore region: it is never a miss, and
      any number of predictions may match it and are then ignored. OKS against
      it is computed like any other target, using its box when it has no
      visible keypoint. Without the key, every target is a regular instance.
    - **Invalid values.** NaN coordinates and a target area of `0` are not
      rejected. As in `pycocotools`, a NaN keypoint adds `0` to the OKS, and
      with a zero area only keypoints at exactly zero distance add to it, so
      such pairs score low or `0`. Clean such labels beforehand.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> from supervision.metrics import KeypointMeanAveragePrecision
        >>> targets = sv.KeyPoints(
        ...     xy=np.array([[[10, 10], [60, 10], [35, 80]]], dtype=np.float32),
        ...     class_id=np.array([0]),
        ...     data={"area": np.array([4000.0])},
        ... )
        >>> predictions = sv.KeyPoints(
        ...     xy=np.array([[[11, 10], [60, 12], [35, 79]]], dtype=np.float32),
        ...     class_id=np.array([0]),
        ...     detection_confidence=np.array([0.9]),
        ... )
        >>> metric = KeypointMeanAveragePrecision(sigmas=[0.25, 0.25, 0.25])
        >>> result = metric.update(predictions, targets).compute()
        >>> round(result.map50, 2)
        1.0

        ```
    """

    def __init__(
        self,
        sigmas: npt.ArrayLike | None = None,
        class_agnostic: bool = False,
    ) -> None:
        """Initialize the keypoint Mean Average Precision metric.

        Args:
            sigmas: Per-keypoint OKS sigmas of shape `(K,)`. Required unless the
                skeleton has 17 keypoints, in which case it defaults to
                the COCO 17-point sigmas. There is no generic default: sigmas
                encode how precisely each keypoint can be annotated, so COCO
                values are wrong for other skeletons.
            class_agnostic: Whether to treat all data as a single class.
        """
        self._sigmas = None if sigmas is None else np.asarray(sigmas, np.float64)
        self._class_agnostic = class_agnostic
        self._num_keypoints: int | None = None
        self._predictions_list: list[KeyPoints] = []
        self._targets_list: list[KeyPoints] = []

    def reset(self) -> None:
        """Reset the metric to its initial state, clearing all stored data."""
        self._num_keypoints = None
        self._predictions_list = []
        self._targets_list = []

    def update(
        self,
        predictions: KeyPoints | list[KeyPoints],
        targets: KeyPoints | list[KeyPoints],
    ) -> KeypointMeanAveragePrecision:
        """Add predictions and targets of one or more images to the metric.

        Args:
            predictions: The predicted key points, one `sv.KeyPoints` per image.
            targets: The ground-truth key points, one `sv.KeyPoints` per image.

        Returns:
            The updated metric instance.

        Raises:
            ValueError: If the numbers of predictions and targets differ, a
                skeleton's number of keypoints differs from earlier ones or
                from the length of `sigmas`, `sigmas` is missing for a skeleton
                that is not 17 points long, `data["xyxy"]` of predictions or
                targets is not of shape `(N, 4)`, or `targets.data["iscrowd"]`
                is not of shape `(N,)`.
        """
        if not isinstance(predictions, list):
            predictions = [predictions]
        if not isinstance(targets, list):
            targets = [targets]

        if len(predictions) != len(targets):
            raise ValueError(
                f"The number of predictions ({len(predictions)}) and"
                f" targets ({len(targets)}) during the update must be the same."
            )

        for key_points in [*predictions, *targets]:
            self._check_num_keypoints(key_points)
        for name, key_points_list in (
            ("predictions", predictions),
            ("targets", targets),
        ):
            for key_points in key_points_list:
                boxes = key_points.data.get(_XYXY_DATA_FIELD)
                if boxes is not None and np.shape(boxes) != (len(key_points), 4):
                    raise ValueError(
                        f"`{name}.data['{_XYXY_DATA_FIELD}']` must have shape "
                        f"({len(key_points)}, 4); got {np.shape(boxes)}."
                    )
        for key_points in targets:
            iscrowd = key_points.data.get(_ISCROWD_DATA_FIELD)
            if iscrowd is not None and np.shape(iscrowd) != (len(key_points),):
                raise ValueError(
                    f"`targets.data['{_ISCROWD_DATA_FIELD}']` must have shape "
                    f"({len(key_points)},); got {np.shape(iscrowd)}."
                )

        self._predictions_list.extend(predictions)
        self._targets_list.extend(targets)
        return self

    def _check_num_keypoints(self, key_points: KeyPoints) -> None:
        """Check that a skeleton matches earlier ones and the given sigmas.

        Args:
            key_points: The skeletons of one image.

        Raises:
            ValueError: If the number of keypoints differs from earlier
                skeletons or from the length of `sigmas`, or `sigmas` is
                missing for a skeleton that is not 17 points long.
        """
        if len(key_points) == 0:
            return
        num_keypoints = key_points.xy.shape[1]
        if self._num_keypoints is None:
            _resolve_keypoint_sigmas(self._sigmas, num_keypoints)
            self._num_keypoints = num_keypoints
        elif num_keypoints != self._num_keypoints:
            raise ValueError(
                f"Skeletons have {num_keypoints} keypoints, but earlier skeletons "
                f"passed to `update` since the last `reset` have "
                f"{self._num_keypoints}; all skeletons must have the same "
                f"number of keypoints."
            )

    def _category_id(self, key_points: KeyPoints, index: int) -> int:
        """Return the evaluation category of one skeleton.

        Args:
            key_points: The skeletons of one image.
            index: The skeleton's index in `key_points`.

        Returns:
            `-1` when class agnostic, `0` without `class_id`, else its class.
        """
        if self._class_agnostic:
            return -1
        if key_points.class_id is None:
            return 0
        return int(key_points.class_id[index])

    def _prepare_targets(
        self,
    ) -> tuple[dict[str, list[_TypeCocoDict]], dict[int, npt.NDArray[np.float64]]]:
        """Transform targets into the COCO dictionary used by the evaluator.

        Returns:
            The COCO dataset dictionary, and the xyxy box by annotation id of
            each kept target without visible keypoints.
        """
        target_boxes: dict[int, npt.NDArray[np.float64]] = {}
        images: list[_TypeCocoDict] = [
            {"id": image_id} for image_id in range(len(self._targets_list))
        ]
        annotations: list[_TypeCocoDict] = []
        for image_id, image_targets in enumerate(self._targets_list):
            if len(image_targets) == 0:
                continue
            # `xy` may carry a third (z) column; OKS is planar.
            xy = image_targets.xy[..., :2].astype(np.float64)
            visible = (
                np.ones(xy.shape[:2], dtype=bool)
                if image_targets.visible is None
                else np.asarray(image_targets.visible, dtype=bool)
            )
            areas = image_targets.data.get(AREA_DATA_FIELD)
            boxes = image_targets.data.get(_XYXY_DATA_FIELD)
            if boxes is not None:
                boxes = np.asarray(boxes, dtype=np.float64)
            iscrowd = image_targets.data.get(_ISCROWD_DATA_FIELD)
            for index in range(len(image_targets)):
                is_ignored = not visible[index].any()
                if is_ignored and boxes is None:
                    continue
                xyxy = boxes[index] if is_ignored and boxes is not None else None
                if areas is not None:
                    area = float(areas[index])
                elif xyxy is not None:
                    area = float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
                else:
                    _, _, width, height = _keypoints_bbox(xy[index][visible[index]])
                    area = width * height
                content = np.column_stack([xy[index], visible[index]])
                annotation_id = len(annotations) + 1  # 0 means no match
                if xyxy is not None:
                    target_boxes[annotation_id] = xyxy
                is_crowd = iscrowd is not None and bool(iscrowd[index])
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": self._category_id(image_targets, index),
                        "area": area,
                        "iscrowd": int(is_crowd),
                        "ignore": int(is_ignored),
                        "content": content,
                    }
                )
        category_ids = sorted({annotation["category_id"] for annotation in annotations})
        categories: list[_TypeCocoDict] = [{"id": cat_id} for cat_id in category_ids]
        dataset = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        return dataset, target_boxes

    def _prepare_predictions(self) -> list[_TypeCocoDict]:
        """Transform predictions into the COCO result list used by the evaluator."""
        coco_predictions: list[_TypeCocoDict] = []
        for image_id, image_predictions in enumerate(self._predictions_list):
            if len(image_predictions) == 0:
                continue
            xy = image_predictions.xy[..., :2].astype(np.float64)
            confidence = image_predictions.detection_confidence
            boxes = image_predictions.data.get(_XYXY_DATA_FIELD)
            for index in range(len(image_predictions)):
                if boxes is None:
                    bbox = _keypoints_bbox(xy[index])
                else:
                    x_min, y_min, x_max, y_max = (float(v) for v in boxes[index])
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                score = 0.0 if confidence is None else float(confidence[index])
                coco_predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": self._category_id(image_predictions, index),
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "score": score,
                        "content": np.column_stack(
                            [xy[index], np.ones(len(xy[index]))]
                        ),
                    }
                )
        return coco_predictions

    def compute(self) -> KeypointMeanAveragePrecisionResult:
        """Compute keypoint mAP from the stored predictions and targets.

        Returns:
            The keypoint Mean Average Precision result.
        """
        sigmas = (
            _resolve_keypoint_sigmas(self._sigmas, self._num_keypoints)
            if self._num_keypoints is not None
            else np.empty(0, dtype=np.float64)
        )
        dataset, target_boxes = self._prepare_targets()
        coco_targets = EvaluationDataset(targets=dataset)
        coco_predictions = coco_targets.load_predictions(self._prepare_predictions())
        evaluator = _KeypointCOCOEvaluator(
            coco_targets, coco_predictions, sigmas, target_boxes
        )
        evaluator.evaluate()

        def make_result(
            size: str,
        ) -> KeypointMeanAveragePrecisionResult:
            """Build the result for one object-size bucket."""
            return KeypointMeanAveragePrecisionResult(
                is_class_agnostic=self._class_agnostic,
                mAP_scores=np.asarray(
                    evaluator.results[f"mAP_scores_{size}"], dtype=np.float64
                ),
                ap_per_class=np.asarray(
                    evaluator.results[f"ap_per_class_{size}"], dtype=np.float64
                ),
                oks_thresholds=np.asarray(evaluator.params.iou_thrs, dtype=np.float64),
                matched_classes=np.asarray(evaluator.params.cat_ids, dtype=np.int32),
            )

        result = make_result("all_sizes")
        result.medium_objects = make_result("medium")
        result.large_objects = make_result("large")
        return result
