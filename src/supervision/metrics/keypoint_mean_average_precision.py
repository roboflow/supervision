from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from supervision.config import AREA_DATA_FIELD, ISCROWD_DATA_FIELD, XYXY_DATA_FIELD
from supervision.detection.utils.iou_and_nms import (
    _keypoint_oks_batch,
    _resolve_keypoint_sigmas,
    _validate_keypoint_sigmas,
)
from supervision.draw.color import LEGACY_COLOR_PALETTE
from supervision.key_points.core import KeyPoints
from supervision.metrics.core import (
    Metric,
    MetricResult,
    PlotDetails,
    _append_object_size_plot_details,
    _mean_valid_score,
    _scores_to_pandas,
    _show_bar_plot,
)
from supervision.metrics.mean_average_precision import (
    _OBJECT_SIZE_AREA_RANGES,
    COCOEvaluator,
    EvaluationDataset,
    ObjectSize,
    _TypeCocoDict,
)

if TYPE_CHECKING:
    import pandas as pd

#: Maximum detections per image and class in COCO keypoint evaluation.
_KEYPOINT_MAX_DETECTIONS = 20

#: Object sizes COCO keypoint evaluation reports, as `pycocotools` `setKpParams`.
_KEYPOINT_OBJECT_SIZES = (ObjectSize.ALL, ObjectSize.MEDIUM, ObjectSize.LARGE)


@dataclass
class KeyPointMeanAveragePrecisionResult(MetricResult):
    """The result of the keypoint Mean Average Precision calculation.

    Scores are `-1` when there are no targets to evaluate. COCO keypoint
    evaluation reports no small-object bucket, because small people are not
    annotated with keypoints in COCO, so only medium and large results exist.

    Attributes:
        is_class_agnostic: When computing class-agnostic results, every
            skeleton gets class ID `-1` when any input has class IDs, and
            otherwise keeps the default class `0`, as in
            `MeanAveragePrecision`.
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
    medium_objects: KeyPointMeanAveragePrecisionResult | None = None
    large_objects: KeyPointMeanAveragePrecisionResult | None = None

    def __str__(self) -> str:
        """Format the scores like the `pycocotools` keypoint summary.

        Example:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> from supervision.metrics import KeyPointMeanAveragePrecision
            >>> xy = np.array([[[10, 10], [60, 10], [35, 80]]], dtype=np.float32)
            >>> targets = sv.KeyPoints(xy=xy, class_id=np.array([0]))
            >>> predictions = sv.KeyPoints(
            ...     xy=xy, class_id=np.array([0]), detection_confidence=np.array([0.9])
            ... )
            >>> metric = KeyPointMeanAveragePrecision(sigmas=[0.5, 0.5, 0.5])
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


class _KeyPointCOCOEvaluator(COCOEvaluator):
    """COCO evaluator that matches by OKS instead of IoU.

    Matching, accumulation and the 101-point interpolation are inherited, so
    keypoint mAP shares them with box, mask and oriented-box mAP. As in
    `pycocotools`, only the all, medium and large area ranges are evaluated:
    small people carry no keypoint labels in COCO, so there is no small bucket
    to compute. Each
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
        """Set up the evaluator with COCO keypoint max detections and area ranges.

        Args:
            coco_targets: The dataset with the ground truths.
            coco_predictions: The dataset with the predictions.
            sigmas: Per-keypoint OKS sigmas.
            target_boxes: xyxy box by annotation id for targets without
                visible keypoints, which OKS measures against.
        """
        super().__init__(coco_targets, coco_predictions)
        self.params.max_dets = [_KEYPOINT_MAX_DETECTIONS]
        self.params.area_range = [
            list(_OBJECT_SIZE_AREA_RANGES[size]) for size in _KEYPOINT_OBJECT_SIZES
        ]
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


def _keypoints_xywh(
    xy: npt.NDArray[np.float64], visible: npt.NDArray[np.bool_] | None = None
) -> npt.NDArray[np.float64]:
    """Return the COCO `[x, y, w, h]` box spanning each skeleton's keypoints.

    Args:
        xy: Keypoint coordinates of shape `(N, K, 2)`, with `K >= 1`.
        visible: Mask of shape `(N, K)` of the keypoints to span; all of them
            when `None`. A skeleton with none gets an infinite box.

    Returns:
        The boxes as `(N, 4)` rows of `[x_min, y_min, width, height]`.
    """
    if visible is None:
        low, high = xy.min(axis=1), xy.max(axis=1)
    else:
        # Masked keypoints are moved past every coordinate, so min and max skip
        # them without a per-skeleton loop.
        mask = visible[..., None]
        low = np.where(mask, xy, np.inf).min(axis=1)
        high = np.where(mask, xy, -np.inf).max(axis=1)
    return np.concatenate([low, high - low], axis=1)


def _check_target_areas(targets: KeyPoints) -> None:
    """Check that `targets.data["area"]` holds one usable area per target.

    Args:
        targets: The ground-truth skeletons of one image.

    Raises:
        ValueError: If the areas are not of shape `(N,)`, or any is negative or
            not finite. An area of `0` is accepted.
    """
    areas = targets.data.get(AREA_DATA_FIELD)
    if areas is None:
        return
    area_values = np.asarray(areas, dtype=np.float64)
    if area_values.shape != (len(targets),):
        raise ValueError(
            f"`targets.data['{AREA_DATA_FIELD}']` must have shape "
            f"({len(targets)},); got {area_values.shape}."
        )
    if not np.all(np.isfinite(area_values) & (area_values >= 0)):
        raise ValueError(
            f"`targets.data['{AREA_DATA_FIELD}']` must be finite and non-negative."
        )


class KeyPointMeanAveragePrecision(Metric[KeyPointMeanAveragePrecisionResult]):
    """Keypoint Mean Average Precision based on Object Keypoint Similarity (OKS).

    This is the COCO keypoint metric: predictions are matched to targets by OKS
    instead of IoU, at 10 OKS thresholds from `0.5` to `0.95`, with 101-point
    recall interpolation and at most 20 predictions per image and class. The
    matching and accumulation are shared with `MeanAveragePrecision`, and the
    result matches `pycocotools` `COCOeval(..., iouType="keypoints")` when the
    target areas match.

    Inputs are one `sv.KeyPoints` per image for predictions and targets:

    - **Visibility.** Target keypoints with `visible` set to `False` are
      unlabelled, like COCO `v=0`, and do not enter the OKS. COCO `v=1`
      (labelled but occluded) and `v=2` (labelled and visible) both map to
      `visible=True`, because `pycocotools` counts every keypoint with `v>0`.
      `visible=None` means every keypoint is labelled. Prediction visibility
      is not used.
    - **Targets without visible keypoints** (COCO `num_keypoints == 0`). When
      `targets.data["xyxy"]` holds the target boxes as `(N, 4)`
      `(x_min, y_min, x_max, y_max)`, such a target is kept as an ignore
      region, as in `pycocotools`: OKS is measured to its box expanded by its
      own width and height on each side, a prediction matched to it is ignored
      rather than counted, and it is never a miss, in every size bucket. Its
      area is `targets.data["area"]` or, if absent, the box area. Without
      boxes the target is skipped, so a prediction that lands on it counts as
      a false positive; if it was the class's only target, the class has no
      targets and scores `-1`. Boxes of targets with visible keypoints are not used.
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
    - **Invalid values.** Non-finite (NaN or infinite) coordinates are not
      rejected. A non-finite target keypoint counts as unlabelled, like
      `visible=False`, so a target with only non-finite keypoints is a target
      without visible keypoints. A non-finite prediction keypoint adds `0` to
      the OKS while its finite keypoints still count. `pycocotools` instead
      returns a NaN OKS when a labelled keypoint of the pair is NaN. A target
      area must be finite and non-negative. With an area of `0` only
      keypoints at exactly zero distance add to the OKS, so such pairs score
      low or `0`. Clean such labels beforehand.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> from supervision.metrics import KeyPointMeanAveragePrecision
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
        >>> metric = KeyPointMeanAveragePrecision(sigmas=[0.25, 0.25, 0.25])
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
            class_agnostic: Whether to treat all data as a single class with ID
                `-1`. As in `MeanAveragePrecision`, when no input has class IDs
                that class keeps the default ID `0`.

        Raises:
            ValueError: If `sigmas` holds a value that is not positive and
                finite. Its length is checked against the skeleton on `update`.
        """
        self._sigmas = None if sigmas is None else np.asarray(sigmas, np.float64)
        if self._sigmas is not None:
            _validate_keypoint_sigmas(self._sigmas)
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
    ) -> KeyPointMeanAveragePrecision:
        """Add predictions and targets of one or more images to the metric.

        Args:
            predictions: The predicted key points, one `sv.KeyPoints` per image.
            targets: The ground-truth key points, one `sv.KeyPoints` per image.

        Returns:
            The updated metric instance.

        Raises:
            ValueError: If the numbers of predictions and targets differ, a
                skeleton has no keypoints, a skeleton's number of keypoints
                differs from earlier ones or from the length of `sigmas`,
                `sigmas` is missing for a skeleton that is not 17 points long,
                `data["xyxy"]` of predictions or targets is not of shape
                `(N, 4)`, `targets.data["iscrowd"]` is not of shape `(N,)`, or
                `targets.data["area"]` is not of shape `(N,)` or holds a
                negative or non-finite area. A rejected update leaves the
                metric unchanged.
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

        num_keypoints = self._num_keypoints
        for key_points in [*predictions, *targets]:
            num_keypoints = self._check_num_keypoints(key_points, num_keypoints)
        for name, key_points_list in (
            ("predictions", predictions),
            ("targets", targets),
        ):
            for key_points in key_points_list:
                boxes = key_points.data.get(XYXY_DATA_FIELD)
                if boxes is not None and np.shape(boxes) != (len(key_points), 4):
                    raise ValueError(
                        f"`{name}.data['{XYXY_DATA_FIELD}']` must have shape "
                        f"({len(key_points)}, 4); got {np.shape(boxes)}."
                    )
        for key_points in targets:
            iscrowd = key_points.data.get(ISCROWD_DATA_FIELD)
            if iscrowd is not None and np.shape(iscrowd) != (len(key_points),):
                raise ValueError(
                    f"`targets.data['{ISCROWD_DATA_FIELD}']` must have shape "
                    f"({len(key_points)},); got {np.shape(iscrowd)}."
                )
            _check_target_areas(key_points)

        # Store anything only once every check passed, so that a rejected update
        # does not pin the skeleton size for later updates.
        self._num_keypoints = num_keypoints
        self._predictions_list.extend(predictions)
        self._targets_list.extend(targets)
        return self

    def _check_num_keypoints(
        self, key_points: KeyPoints, num_keypoints: int | None
    ) -> int | None:
        """Check that a skeleton matches earlier ones and the given sigmas.

        Args:
            key_points: The skeletons of one image.
            num_keypoints: The number of keypoints of earlier skeletons, or
                `None` if there were none.

        Returns:
            The number of keypoints of these and earlier skeletons, or `None`
            while there are none.

        Raises:
            ValueError: If the skeletons have no keypoints, the number of
                keypoints differs from earlier skeletons or from the length of
                `sigmas`, or `sigmas` is missing for a skeleton that is not 17
                points long.
        """
        if len(key_points) == 0:
            return num_keypoints
        skeleton_size = int(key_points.xy.shape[1])
        if skeleton_size == 0:
            raise ValueError(
                "Skeletons must have at least one keypoint; got `xy` of shape "
                f"{key_points.xy.shape}."
            )
        if num_keypoints is None:
            _resolve_keypoint_sigmas(self._sigmas, skeleton_size)
            return skeleton_size
        if skeleton_size != num_keypoints:
            raise ValueError(
                f"Skeletons have {skeleton_size} keypoints, but earlier skeletons "
                f"passed to `update` since the last `reset` have "
                f"{num_keypoints}; all skeletons must have the same "
                f"number of keypoints."
            )
        return num_keypoints

    def _agnostic_category_id(self) -> int:
        """Return the single category of all skeletons when class agnostic.

        As in `MeanAveragePrecision`, it is `-1` when any stored skeleton has a
        class ID, and otherwise the default class `0`.
        """
        has_class_ids = any(
            key_points.class_id is not None and len(key_points) > 0
            for key_points in [*self._predictions_list, *self._targets_list]
        )
        return -1 if has_class_ids else 0

    def _category_id(self, key_points: KeyPoints, index: int, agnostic_id: int) -> int:
        """Return the evaluation category of one skeleton.

        Args:
            key_points: The skeletons of one image.
            index: The skeleton's index in `key_points`.
            agnostic_id: The category of every skeleton when class agnostic.

        Returns:
            `agnostic_id` when class agnostic, `0` without `class_id`, else its
            class.
        """
        if self._class_agnostic:
            return agnostic_id
        if key_points.class_id is None:
            return 0
        return int(key_points.class_id[index])

    def _prepare_targets(
        self, agnostic_id: int
    ) -> tuple[dict[str, list[_TypeCocoDict]], dict[int, npt.NDArray[np.float64]]]:
        """Transform targets into the COCO dictionary used by the evaluator.

        Args:
            agnostic_id: The category of every target when class agnostic.

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
            # A non-finite target keypoint is unlabelled, like COCO `v=0`, so it
            # stays out of the OKS and the fallback area. Not in place: `visible`
            # may be the caller's array.
            visible = visible & np.isfinite(xy).all(axis=-1)
            areas = image_targets.data.get(AREA_DATA_FIELD)
            boxes = image_targets.data.get(XYXY_DATA_FIELD)
            if boxes is not None:
                boxes = np.asarray(boxes, dtype=np.float64)
            iscrowd = image_targets.data.get(ISCROWD_DATA_FIELD)
            # Per-image arrays, so that the loop below only picks rows. Rows of
            # targets without visible keypoints are infinite and never read.
            span_boxes = _keypoints_xywh(xy, visible)
            span_areas = span_boxes[:, 2] * span_boxes[:, 3]
            contents = np.concatenate([xy, visible[..., None]], axis=2)
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
                    area = float(span_areas[index])
                content = contents[index]
                annotation_id = len(annotations) + 1  # 0 means no match
                if xyxy is not None:
                    target_boxes[annotation_id] = xyxy
                is_crowd = iscrowd is not None and bool(iscrowd[index])
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": self._category_id(
                            image_targets, index, agnostic_id
                        ),
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

    def _prepare_predictions(self, agnostic_id: int) -> list[_TypeCocoDict]:
        """Transform predictions into the COCO result list used by the evaluator.

        Args:
            agnostic_id: The category of every prediction when class agnostic.
        """
        coco_predictions: list[_TypeCocoDict] = []
        for image_id, image_predictions in enumerate(self._predictions_list):
            if len(image_predictions) == 0:
                continue
            xy = image_predictions.xy[..., :2].astype(np.float64)
            confidence = image_predictions.detection_confidence
            boxes = image_predictions.data.get(XYXY_DATA_FIELD)
            # Per-image arrays, so that the loop below only picks rows.
            if boxes is None:
                bboxes = _keypoints_xywh(xy)
            else:
                xyxy = np.asarray(boxes, dtype=np.float64)
                bboxes = np.concatenate([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]], 1)
            scores = (
                np.zeros(len(image_predictions))
                if confidence is None
                else np.asarray(confidence, dtype=np.float64)
            )
            areas = bboxes[:, 2] * bboxes[:, 3]
            contents = np.concatenate([xy, np.ones((*xy.shape[:2], 1))], axis=2)
            rows = zip(bboxes.tolist(), areas.tolist(), scores.tolist())
            for index, (bbox, area, score) in enumerate(rows):
                coco_predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": self._category_id(
                            image_predictions, index, agnostic_id
                        ),
                        "bbox": bbox,
                        "area": area,
                        "score": score,
                        "content": contents[index],
                    }
                )
        return coco_predictions

    def compute(self) -> KeyPointMeanAveragePrecisionResult:
        """Compute keypoint mAP from the stored predictions and targets.

        Returns:
            The keypoint Mean Average Precision result.
        """
        sigmas = (
            _resolve_keypoint_sigmas(self._sigmas, self._num_keypoints)
            if self._num_keypoints is not None
            else np.empty(0, dtype=np.float64)
        )
        agnostic_id = self._agnostic_category_id()
        dataset, target_boxes = self._prepare_targets(agnostic_id)
        coco_targets = EvaluationDataset(targets=dataset)
        coco_predictions = coco_targets.load_predictions(
            self._prepare_predictions(agnostic_id)
        )
        evaluator = _KeyPointCOCOEvaluator(
            coco_targets, coco_predictions, sigmas, target_boxes
        )
        evaluator.evaluate()

        def make_result(
            size: str,
        ) -> KeyPointMeanAveragePrecisionResult:
            """Build the result for one object-size bucket."""
            return KeyPointMeanAveragePrecisionResult(
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
