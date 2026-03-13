from __future__ import annotations

import copy
import datetime
import itertools
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

from supervision.detection.core import Detections
from supervision.detection.utils.iou_and_nms import box_iou_batch_with_jaccard
from supervision.draw.color import LEGACY_COLOR_PALETTE
from supervision.metrics.core import Metric, MetricTarget
from supervision.metrics.utils.utils import ensure_pandas_installed
from supervision.utils.logger import _get_logger

logger = _get_logger(__name__)

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class MeanAveragePrecisionResult:
    """
    The result of the Mean Average Precision calculation.

    Defaults to `0` when no detections or targets are present.

    Attributes:
        metric_target: the type of data used for the metric -
            boxes, masks or oriented bounding boxes.
        is_class_agnostic: When computing class-agnostic results, class ID
            is set to `-1`.
        mAP_scores: the mAP scores at each IoU threshold.
            Shape: `(num_iou_thresholds,)`
        ap_per_class: the average precision scores per
            class and IoU threshold. Shape: `(num_target_classes, num_iou_thresholds)`
        iou_thresholds: the IoU thresholds used in the calculations.
        matched_classes: the class IDs of all matched classes.
            Corresponds to the rows of `ap_per_class`.
        small_objects: the mAP results
            for small objects (area < 32²).
        medium_objects: the mAP results
            for medium objects (32² ≤ area < 96²).
        large_objects: the mAP results
            for large objects (area ≥ 96²).
    """

    metric_target: MetricTarget
    is_class_agnostic: bool

    @property
    def map50_95(self) -> float:
        """the mAP score at IoU thresholds from `0.5` to `0.95`."""
        valid_scores = self.mAP_scores[self.mAP_scores > -1]
        if len(valid_scores) > 0:
            return float(valid_scores.mean())
        else:
            return -1

    @property
    def map50(self) -> float:
        """the mAP score at IoU threshold of `0.5`."""
        return float(self.mAP_scores[0])

    @property
    def map75(self) -> float:
        """the mAP score at IoU threshold of `0.75`."""
        return float(self.mAP_scores[5])

    mAP_scores: npt.NDArray[np.float64]
    ap_per_class: npt.NDArray[np.float64]
    iou_thresholds: npt.NDArray[np.float64]
    matched_classes: npt.NDArray[np.int32]
    small_objects: MeanAveragePrecisionResult | None = None
    medium_objects: MeanAveragePrecisionResult | None = None
    large_objects: MeanAveragePrecisionResult | None = None

    def __str__(self) -> str:
        """
        Formats the evaluation output metrics to match the structure used by pycocotools

        Example:
           ```pycon
           >>> import numpy as np
           >>> import supervision as sv
           >>> from supervision.metrics import MeanAveragePrecision
           >>> predictions = sv.Detections(
           ...     xyxy=np.array([[0, 0, 10, 10]]),
           ...     class_id=np.array([0]),
           ...     confidence=np.array([0.9])
           ... )
           >>> targets = sv.Detections(
           ...     xyxy=np.array([[0, 0, 10, 10]]),
           ...     class_id=np.array([0])
           ... )
           >>> map_metric = MeanAveragePrecision()
           >>> map_result = map_metric.update(predictions, targets).compute()
           >>> print(map_result)  # doctest: +ELLIPSIS
           Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ...
           Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = ...
           Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = ...
           Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = ...
           Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = ...
           Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = ...

           ```
        """
        if (
            self.small_objects is None
            or self.medium_objects is None
            or self.large_objects is None
        ):
            return (
                f"Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | "
                f"maxDets=100 ] = {self.map50_95:.3f}\n"
                f"Average Precision (AP) @[ IoU=0.50      | area=   all | "
                f"maxDets=100 ] = {self.map50:.3f}\n"
                f"Average Precision (AP) @[ IoU=0.75      | area=   all | "
                f"maxDets=100 ] = {self.map75:.3f}"
            )

        return (
            f"Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | "
            f"maxDets=100 ] = {self.map50_95:.3f}\n"
            f"Average Precision (AP) @[ IoU=0.50      | area=   all | "
            f"maxDets=100 ] = {self.map50:.3f}\n"
            f"Average Precision (AP) @[ IoU=0.75      | area=   all | "
            f"maxDets=100 ] = {self.map75:.3f}\n"
            f"Average Precision (AP) @[ IoU=0.50:0.95 | area= small | "
            f"maxDets=100 ] = {self.small_objects.map50_95:.3f}\n"
            f"Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | "
            f"maxDets=100 ] = {self.medium_objects.map50_95:.3f}\n"
            f"Average Precision (AP) @[ IoU=0.50:0.95 | area= large | "
            f"maxDets=100 ] = {self.large_objects.map50_95:.3f}"
        )

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert the result to a pandas DataFrame.

        Returns:
            The result as a DataFrame.
        """
        ensure_pandas_installed()
        import pandas as pd

        pandas_data = {
            "mAP@50:95": self.map50_95,
            "mAP@50": self.map50,
            "mAP@75": self.map75,
        }

        if self.small_objects is not None:
            small_objects_df = self.small_objects.to_pandas()
            for key, value in small_objects_df.items():
                pandas_data[f"small_objects_{key}"] = value
        if self.medium_objects is not None:
            medium_objects_df = self.medium_objects.to_pandas()
            for key, value in medium_objects_df.items():
                pandas_data[f"medium_objects_{key}"] = value
        if self.large_objects is not None:
            large_objects_df = self.large_objects.to_pandas()
            for key, value in large_objects_df.items():
                pandas_data[f"large_objects_{key}"] = value

        # Average precisions are currently not included in the DataFrame.
        return pd.DataFrame(
            pandas_data,
            index=[0],
        )

    def plot(self) -> None:
        """
        Plot the mAP results.

        ![example_plot](
            https://media.roboflow.com/supervision-docs/metrics/mAP_plot_example.png
        ){ align=center width="800" }
        """

        labels = ["mAP@50:95", "mAP@50", "mAP@75"]
        values = [self.map50_95, self.map50, self.map75]
        colors = [LEGACY_COLOR_PALETTE[0]] * 3

        if self.small_objects is not None:
            labels += ["Small: mAP@50:95", "Small: mAP@50", "Small: mAP@75"]
            values += [
                self.small_objects.map50_95,
                self.small_objects.map50,
                self.small_objects.map75,
            ]
            colors += [LEGACY_COLOR_PALETTE[3]] * 3

        if self.medium_objects is not None:
            labels += ["Medium: mAP@50:95", "Medium: mAP@50", "Medium: mAP@75"]
            values += [
                self.medium_objects.map50_95,
                self.medium_objects.map50,
                self.medium_objects.map75,
            ]
            colors += [LEGACY_COLOR_PALETTE[2]] * 3

        if self.large_objects is not None:
            labels += ["Large: mAP@50:95", "Large: mAP@50", "Large: mAP@75"]
            values += [
                self.large_objects.map50_95,
                self.large_objects.map50,
                self.large_objects.map75,
            ]
            colors += [LEGACY_COLOR_PALETTE[4]] * 3

        plt.rcParams["font.family"] = "monospace"

        _, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylim(0, 1)
        ax.set_ylabel("Value", fontweight="bold")
        ax.set_title("Mean Average Precision", fontweight="bold")

        x_positions = range(len(labels))
        bars = ax.bar(x_positions, values, color=colors, align="center")

        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")

        for bar in bars:
            y_value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_value + 0.02,
                f"{y_value:.2f}",
                ha="center",
                va="bottom",
            )

        plt.rcParams["font.family"] = "sans-serif"

        plt.tight_layout()
        plt.show()


class EvaluationDataset:
    """
    Class used representing a dataset in the right format needed by the
    `COCOEvaluator` class.
    """

    def __init__(self, targets: dict[str, Any] | None = None):
        """
        Constructor of EvaluationDataset object used to evaluate models with
        Mean Average Precision.

        Args:
            targets: The targets (ground truth) of the dataset in a the
                COCO format.
        """
        # Initialize members
        # Initialize members
        self.dataset: dict[str, Any] = dict()
        self.anns: dict[int, Any] = dict()
        self.cats: dict[int, Any] = dict()
        self.imgs: dict[int, Any] = dict()
        self.img_to_anns: dict[int, list[Any]] = defaultdict(list)
        self.cat_to_imgs: dict[int, list[int]] = defaultdict(list)

        if targets is None:
            return

        # Load dataset
        self.dataset = targets
        self.create_class_members()

    @classmethod
    def empty(cls) -> EvaluationDataset:
        return cls(targets=None)

    def create_class_members(self) -> None:
        """
        Create index elements for the dataset.
        """
        anns, cats, imgs = {}, {}, {}
        img_to_anns, cat_to_imgs = defaultdict(list), defaultdict(list)
        if "annotations" in self.dataset:
            for ann in self.dataset["annotations"]:
                img_to_anns[ann["image_id"]].append(ann)
                anns[ann["id"]] = ann

        if "images" in self.dataset:
            for img in self.dataset["images"]:
                imgs[img["id"]] = img

        if "categories" in self.dataset:
            for cat in self.dataset["categories"]:
                cats[cat["id"]] = cat

        if "annotations" in self.dataset and "categories" in self.dataset:
            for ann in self.dataset["annotations"]:
                cat_to_imgs[ann["category_id"]].append(ann["image_id"])

        # Populate class members
        self.anns = anns
        self.img_to_anns = img_to_anns
        self.cat_to_imgs = cat_to_imgs
        self.imgs = imgs
        self.cats = cats

    def get_annotation_ids(
        self,
        img_ids: list[int] | None = None,
        cat_ids: list[int] | None = None,
        area_range: tuple[float, float] | None = None,
        iscrowd: bool = False,
    ) -> list[int]:
        """
        Get annotation ids that satisfy given filter conditions.

        Args:
            img_ids: ids of the images that we want to retrieve.
            cat_ids: ids of the categories that we want to retrieve.
            area_range: area range of the annotations that we want to retrieve
                in the format [min_area, max_area].
            iscrowd: if annotations to retrieve are `iscrowded=1`.
        """
        # If there are no filters, we use all annotations
        if not img_ids and not cat_ids and not area_range:
            anns = self.dataset["annotations"]
        else:
            if img_ids:
                lists = [
                    self.img_to_anns[img_id]
                    for img_id in img_ids
                    if img_id in self.img_to_anns
                ]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset["annotations"]

            # Filter by category
            anns = (
                anns
                if not cat_ids
                else [ann for ann in anns if ann["category_id"] in cat_ids]
            )

            # Filter by area
            anns = (
                anns
                if not area_range
                else [
                    ann for ann in anns if area_range[0] < ann["area"] < area_range[1]
                ]
            )

        # Filter by iscrowd
        if iscrowd is True:
            ids = [ann["id"] for ann in anns if ann["iscrowd"] == 1]
        else:
            ids = [ann["id"] for ann in anns]
        return ids

    def get_category_ids(
        self,
        cat_names: list[str] | None = None,
        supercategory_names: list[str] | None = None,
        cat_ids: list[int] | None = None,
    ) -> list[int]:
        """
        Get category ids that satisfy given filter conditions.

        Args:
            cat_names: names of the categories to retrieve.
            supercategory_names: names of the supercategories to retrieve.
            cat_ids: ids of the categories to retrieve.

        Returns:
            ids: integer array of category ids.
        """
        # If there are no filters, we use all categories
        if not cat_names and not supercategory_names and not cat_ids:
            cats = self.dataset["categories"]
        else:
            cats = self.dataset["categories"]

            # Filter by name
            cats = (
                cats
                if not cat_names
                else [cat for cat in cats if cat["name"] in cat_names]
            )

            # Filter by supercategory
            cats = (
                cats
                if not supercategory_names
                else [
                    cat for cat in cats if cat["supercategory"] in supercategory_names
                ]
            )

            # Filter by id
            cats = (
                cats if not cat_ids else [cat for cat in cats if cat["id"] in cat_ids]
            )
        ids = [cat["id"] for cat in cats]
        return ids

    def get_image_ids(
        self,
        img_ids: list[int] | None = None,
        cat_ids: list[int] | None = None,
    ) -> list[int]:
        """
        Get image ids that satisfy given filter conditions.

        Args:
            img_ids: ids of the images to retrieve.
            cat_ids: ids of the categories to retrieve.

        Returns:
            ids: integer array of image ids.
        """
        # If there are no filters, we use all images
        if not img_ids and not cat_ids:
            ids = self.imgs.keys()
            return list(ids)

        ids_set = set(img_ids) if img_ids else set()

        if cat_ids:
            for i, cat_id in enumerate(cat_ids):
                if i == 0 and not ids_set:
                    ids_set = set(self.cat_to_imgs[cat_id])
                else:
                    ids_set &= set(self.cat_to_imgs[cat_id])

        return list(ids_set)

    def get_annotations(self, ids: list[int] | None = None) -> list[dict[str, Any]]:
        """
        Get annotations with the specified ids.

        Args:
            ids: integer ids specifying annotations.

        Returns:
            anns: loaded annotations.
        """
        if ids is None:
            return []
        return [self.anns[idx] for idx in ids]

    def load_predictions(self, predictions: list[dict[str, Any]]) -> EvaluationDataset:
        """
        Load prediction result into an EvaluationDataset object.

        Args:
            predictions: prediction result.

        Returns:
            EvaluationDataset object representing the predictions.
        """
        # Create an empty EvaluationDataset object for the predictions
        predictions_dataset = EvaluationDataset.empty()
        predictions_dataset.dataset["images"] = [img for img in self.dataset["images"]]

        if not isinstance(predictions, list):
            raise ValueError("results must be a list")

        # Handle empty predictions
        if len(predictions) == 0:
            predictions_dataset.dataset["annotations"] = []
            return predictions_dataset

        ids = [pred["image_id"] for pred in predictions]

        # Make sure the image ids from predictions exist in the current dataset
        assert set(ids) == (set(ids) & set(self.get_image_ids())), (
            "Results do not correspond to current coco set"
        )

        # Check if the predictions contain any unsupported keys
        if "caption" in predictions[0]:
            raise NotImplementedError(
                "Evaluating predictions with caption is not supported."
            )
        elif "segmentation" in predictions[0]:
            raise NotImplementedError(
                "Evaluating predictions with segmentation is not supported."
            )
        elif "keypoints" in predictions[0]:
            raise NotImplementedError(
                "Evaluating predictions with keypoints is not supported."
            )

        elif "bbox" in predictions[0] and not predictions[0]["bbox"] == []:
            predictions_dataset.dataset["categories"] = copy.deepcopy(
                self.dataset["categories"]
            )

            # Prepare fields for every prediction of the given image
            for idx, pred in enumerate(predictions):
                x, y, w, h = pred["bbox"]
                x1, x2, y1, y2 = [x, x + w, y, y + h]

                # Make segmentation from bounding box coordinates
                if "segmentation" not in pred:
                    pred["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                # Use provided area if available
                if "area" not in pred:
                    pred["area"] = w * h
                pred["id"] = idx + 1
                # For predictions we set iscrowd to 0
                pred["iscrowd"] = 0
        predictions_dataset.dataset["annotations"] = predictions
        predictions_dataset.create_class_members()
        return predictions_dataset


# Area ranges for object size in pixels
SMALL_OBJECT_AREA = 32**2
MEDIUM_OBJECT_AREA = 96**2
MAX_ALL_OBJECT_AREA = 1e5**2

# Smallest number to avoid division by zero
EPS = np.finfo(np.float32).eps


class ObjectSize(Enum):
    """
    Enum for object size.
    """

    ALL = "all"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class COCOEvaluatorParameters:
    """
    Parameters for COCOEvaluator
    """

    def __init__(self) -> None:
        """Initialize all parameters for evaluation"""

        self.img_ids: list[int] = []
        self.cat_ids: list[int] = []

        # IoU thresholds [0.5, 0.55, 0.6, 0.65, ..., 0.95]
        self.iou_thrs = np.linspace(
            0.5,
            0.95,
            int(np.round((0.95 - 0.5) / 0.05)) + 1,
            endpoint=True,
            dtype=np.float32,
        )
        # 101 recall thresholds [0.0, 0.01, 0.02, ..., 1.00]
        self.rec_thrs = np.linspace(
            0.0,
            1.00,
            int(np.round((1.00 - 0.0) / 0.01)) + 1,
            endpoint=True,
            dtype=np.float32,
        )
        # 3 maximum detection thresholds [1, 10, 100]
        self.max_dets = [1, 10, 100]
        # Area ranges [0, 1e5], [0, 32], [32, 96], [96, 1e5]
        self.area_range: list[list[float]] = [
            [0, MAX_ALL_OBJECT_AREA],
            [0, SMALL_OBJECT_AREA],
            [SMALL_OBJECT_AREA, MEDIUM_OBJECT_AREA],
            [MEDIUM_OBJECT_AREA, MAX_ALL_OBJECT_AREA],
        ]


class COCOEvaluator:
    """
    Evaluator class to compute COCO metrics.
    """

    def __init__(
        self, coco_targets: EvaluationDataset, coco_predictions: EvaluationDataset
    ):
        """
        Constructor of COCOEvaluator object.

        Args:
            coco_targets: The dataset with the ground truths.
            coco_predictions: The dataset with the predictions.
        """
        if coco_targets is None:
            raise ValueError("coco_targets must be provided")
        if coco_predictions is None:
            raise ValueError("coco_predictions must be provided")

        self.coco_targets = coco_targets
        self.coco_predictions = coco_predictions
        # List of dictionaries containing the evaluation results
        # len(eval_imgs) = (categories) * (area_ranges) * (images)
        # For COCO 2017: len(eval_images) = 80 * 4 * 5000 = 1600000
        self.eval_imgs: Any = defaultdict(list)
        # Dictionary of accumulated results
        self.results: dict[str, Any] = {}
        # Dictionary of targets for evaluation
        self._targets: defaultdict[tuple[int, int], list[Any]] = defaultdict(list)
        self._predictions: defaultdict[tuple[int, int], list[Any]] = defaultdict(list)
        # Parameters for evaluation
        self.params = COCOEvaluatorParameters()
        # List of results summarization
        self.stats: list[Any] = []
        # Dictionary of IOUs between all targets and predictions
        self.ious: dict[tuple[int, int], Any] = {}
        # Set image and category ids
        self.params.img_ids = sorted(self.coco_targets.get_image_ids())
        self.params.cat_ids = sorted(self.coco_targets.get_category_ids())

    def _prepare_targets_and_predictions(self) -> None:
        """
        Prepare targets and predictions for evaluation.
        """
        # Get the target samples for the evaluation
        annotation_ids = self.coco_targets.get_annotation_ids(
            img_ids=self.params.img_ids, cat_ids=self.params.cat_ids
        )
        targets = self.coco_targets.get_annotations(annotation_ids)
        # Get the prediction samples for the evaluation
        prediction_ids = self.coco_predictions.get_annotation_ids(
            img_ids=self.params.img_ids, cat_ids=self.params.cat_ids
        )
        predictions = self.coco_predictions.get_annotations(prediction_ids)

        # Set ignore flag
        for gt in targets:
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]

        # Select targets
        self._targets = defaultdict(list)
        for gt in targets:
            self._targets[gt["image_id"], gt["category_id"]].append(gt)

        # Select predictions
        self._predictions = defaultdict(list)
        for dt in predictions:
            self._predictions[dt["image_id"], dt["category_id"]].append(dt)

        # Initialize evaluation results
        self.eval_imgs = defaultdict(list)
        self.results = {}

    def _compute_iou(self, img_id: int, cat_id: int) -> npt.NDArray[np.float32]:
        """
        Compute the IoU between the targets and predictions for a given image and
        category.

        Args:
            img_id: The image id.
            cat_id: The category id.

        Returns:
            The IoU between the targets and predictions.
        """

        gt = self._targets[img_id, cat_id]
        dt = self._predictions[img_id, cat_id]

        # If there is nothing to evaluate
        if len(gt) == 0 and len(dt) == 0:
            empty_result: npt.NDArray[np.float32] = np.array([], dtype=np.float32)
            return empty_result

        # Sort predictions by highest score first
        inds = np.argsort([-d["score"] for d in dt], kind="stable")
        dt = [dt[i] for i in inds]

        # Truncate the predictions if there are more predictions than the max detections
        # to evaluate
        if len(dt) > self.params.max_dets[-1]:
            dt = dt[0 : self.params.max_dets[-1]]

        gt_boxes = [g["bbox"] for g in gt]
        dt_boxes = [d["bbox"] for d in dt]

        # Get the iscrowd flag for each gt
        is_crowd = [bool(o["iscrowd"]) for o in gt]
        # Compute iou between each prediction a and gt region
        iou = box_iou_batch_with_jaccard(gt_boxes, dt_boxes, is_crowd).astype(
            np.float32
        )
        return iou

    def _evaluate_image(
        self,
        img_id: int,
        cat_id: int,
        area_range: list[float] | tuple[float, float],
        max_det: int,
    ) -> dict[str, Any] | None:
        """
        Perform evaluation for single category and image.
        Args:
            img_id: The image id.
            cat_id: The category id.
            area_range: The area range.
            max_det: The maximum number of detections.

        Returns:
            The evaluation results.
        """
        # Get targets (gt) and predictions (dt) for the given image and category
        gt: list[dict[str, Any]] = self._targets[img_id, cat_id]
        dt: list[dict[str, Any]] = self._predictions[img_id, cat_id]

        # If there is nothing to evaluate
        if len(gt) == 0 and len(dt) == 0:
            return None

        min_area, max_area = area_range

        # Create an `_ignore` flag for targets if they are set as ignore or their area
        # is not in the range [min_area, max_area]
        for g in gt:
            if g["ignore"] or not (min_area <= g["area"] <= max_area):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        # Sort ground-truths by ignore flag (0: non ignored, 1: ignored)
        gt_sorted = np.argsort([g["_ignore"] for g in gt], kind="stable")
        gt = [gt[i] for i in gt_sorted]

        # Sort predictions by scores in descending order
        dt_sorted = np.argsort([-d["score"] for d in dt], kind="stable")
        dt = [dt[i] for i in dt_sorted[0:max_det]]

        # Load computed ious for the given image and category
        ious = (
            self.ious[img_id, cat_id][:, gt_sorted]
            if len(self.ious[img_id, cat_id]) > 0
            else self.ious[img_id, cat_id]
        )

        # Get the number of thresholds, ground truths and detections
        num_thresholds = len(self.params.iou_thrs)
        num_ground_truths = len(gt)
        num_detections = len(dt)

        # Initialize matches: 0 means no match
        gt_matches = np.zeros((num_thresholds, num_ground_truths))
        dt_matches = np.zeros((num_thresholds, num_detections))
        # Initialize ignore flags: 0 means no ignore
        gt_ignore = np.array([g["_ignore"] for g in gt])
        dt_ignore = np.zeros((num_thresholds, num_detections))
        if len(ious) != 0:
            # Go through the iou thresholds
            for tresh_idx, thresh in enumerate(self.params.iou_thrs):
                # Go through the detections
                for det_idx, det in enumerate(dt):
                    # Start the iou of the best match
                    iou_best_match = min([thresh, 1 - 1e-10])
                    # Set the best match index to -1 (unmatched)
                    best_match_idx = -1
                    # Go through the ground truths
                    for g_idx, g in enumerate(gt):
                        # If current gt is already matched, and not a crowd, continue
                        # if gt_matches[tresh_idx, g_idx] > 0 and not iscrowd[g_idx]:
                        iscrowd = int(g.get("iscrowd", 0))
                        if gt_matches[tresh_idx, g_idx] > 0 and not iscrowd:
                            continue
                        # Stop searching the ground truths
                        if (
                            best_match_idx > -1  # detection is matched to a gt
                            and gt_ignore[best_match_idx]
                            == 0  # matched gt is not ignored
                            and gt_ignore[g_idx] == 1  # current gt is ignored
                        ):
                            break

                        # A new best match was found
                        if ious[det_idx, g_idx] >= iou_best_match:
                            iou_best_match = ious[det_idx, g_idx]
                            best_match_idx = g_idx

                    # A best match was found
                    if best_match_idx != -1:
                        dt_ignore[tresh_idx, det_idx] = gt_ignore[best_match_idx]
                        dt_matches[tresh_idx, det_idx] = gt[best_match_idx]["id"]
                        gt_matches[tresh_idx, best_match_idx] = det["id"]

        # Set unmatched detections outside of area range to ignore
        area_range_mask = np.array(
            [d["area"] < min_area or d["area"] > max_area for d in dt]
        ).reshape((1, len(dt)))

        # Update the ignore flags for detections
        dt_ignore = np.logical_or(
            dt_ignore,
            np.logical_and(
                dt_matches == 0, np.repeat(area_range_mask, num_thresholds, 0)
            ),
        )

        return {
            "image_id": img_id,
            "category_id": cat_id,
            "area_range": area_range,
            "max_det": max_det,
            "dt_ids": [d["id"] for d in dt],
            "gt_ids": [g["id"] for g in gt],
            "dtMatches": dt_matches,
            "gtMatches": gt_matches,
            "dtScores": [d["score"] for d in dt],
            "gtIgnore": gt_ignore,
            "dtIgnore": dt_ignore,
        }

    def _accumulate(self) -> None:
        """
        Accumulate per image evaluation results and store the result in self.results
        """
        # Get the number of thresholds, categories, area ranges, and max detections
        num_iou_thresholds = len(self.params.iou_thrs)
        num_recall_thresholds = len(self.params.rec_thrs)
        num_categories = len(self.params.cat_ids)
        num_area_ranges = len(self.params.area_range)
        num_max_detections = len(self.params.max_dets)
        num_imgs = len(self.params.img_ids)

        # Initialize precision, recall, and scores arrays
        # -1 means absent categories
        precision = -np.ones(
            (
                num_iou_thresholds,
                num_recall_thresholds,
                num_categories,
                num_area_ranges,
                num_max_detections,
            ),
            dtype=np.float32,
        )
        recall = -np.ones(
            (num_iou_thresholds, num_categories, num_area_ranges, num_max_detections),
            dtype=np.float32,
        )
        scores = -np.ones(
            (
                num_iou_thresholds,
                num_recall_thresholds,
                num_categories,
                num_area_ranges,
                num_max_detections,
            ),
            dtype=np.float32,
        )

        # Create sets for indexing
        set_categories = set(self.params.cat_ids)
        set_area_ranges: set[tuple[float, ...]] = {
            tuple(a) for a in self.params.area_range
        }
        set_max_detections = set(self.params.max_dets)
        set_image_ids = set(self.params.img_ids)

        # Select category indexes to evaluate
        selected_category_ids = [
            n for n, k in enumerate(self.params.cat_ids) if k in set_categories
        ]
        # Select max detections to evaluate
        selected_max_detections = [
            m for m in self.params.max_dets if m in set_max_detections
        ]
        # Select area ranges to evaluate
        selected_area_ranges_ids = [
            idx
            for idx, area in enumerate(self.params.area_range)
            if tuple(area) in set_area_ranges
        ]
        # Select image indexes to evaluate
        image_inds = [
            n for n, i in enumerate(self.params.img_ids) if i in set_image_ids
        ]

        # Evaluating at all categories, area ranges, max number of detections, and
        # IoU thresholds

        # Loop through categories
        for cat_idx, cat_eval_idx in enumerate(selected_category_ids):
            cat_offset = cat_eval_idx * num_area_ranges * num_imgs

            # Loop through area ranges
            for area_idx, area_eval_idx in enumerate(selected_area_ranges_ids):
                area_offset = area_eval_idx * num_imgs

                # Loop through max detections
                for max_det_idx, max_det in enumerate(selected_max_detections):
                    eval_img_data = [
                        self.eval_imgs[cat_offset + area_offset + i] for i in image_inds
                    ]
                    eval_img_data = [e for e in eval_img_data if e is not None]

                    # No image to evaluate
                    if len(eval_img_data) == 0:
                        continue

                    # Sort detected scores in descending order
                    dt_scores = np.concatenate(
                        [e["dtScores"][0:max_det] for e in eval_img_data]
                    )
                    inds = np.argsort(-dt_scores, kind="stable")
                    dt_scores_sorted = dt_scores[inds]

                    # Get matches and ignored matches
                    dt_matches = np.concatenate(
                        [e["dtMatches"][:, 0:max_det] for e in eval_img_data], axis=1
                    )[:, inds]
                    dt_ignored = np.concatenate(
                        [e["dtIgnore"][:, 0:max_det] for e in eval_img_data], axis=1
                    )[:, inds]

                    # Get ignored ground truth objects
                    gt_ignored = np.concatenate([e["gtIgnore"] for e in eval_img_data])
                    num_non_ignored_gt = np.count_nonzero(gt_ignored == 0)

                    # No ground truth objects to evaluate
                    if num_non_ignored_gt == 0:
                        continue

                    # Compute true positives and false positives
                    true_positives = np.logical_and(
                        dt_matches, np.logical_not(dt_ignored)
                    )
                    false_positives = np.logical_and(
                        np.logical_not(dt_matches), np.logical_not(dt_ignored)
                    )

                    tp_sum = np.cumsum(true_positives, axis=1).astype(dtype=np.float32)
                    fp_sum = np.cumsum(false_positives, axis=1).astype(dtype=np.float32)

                    # Loop through thresholds
                    for iou_thresh_idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        num_tps = len(tp)
                        # Recall: TP / Total number of ground truth objects
                        rc = tp / np.float32(num_non_ignored_gt)
                        # Precision: TP / (FP + TP)
                        pr = (tp / (fp + tp + EPS)).tolist()
                        # List to compute the precision at each recall threshold
                        precision_at_recall = [0.0] * num_recall_thresholds
                        # List to compute the score at each recall threshold
                        score_at_recall = [0.0] * num_recall_thresholds

                        # Set recall to either the final recall value or 0 (when there
                        # is no TP)
                        recall[iou_thresh_idx, cat_idx, area_idx, max_det_idx] = (
                            rc[-1] if num_tps else 0
                        )

                        # Loop through precision values
                        for i in range(num_tps - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        recall_inds: npt.NDArray[np.int_] = np.searchsorted(
                            rc, self.params.rec_thrs, side="left"
                        )
                        recall_inds_list: list[int] = recall_inds.tolist()
                        for ri, pos_idx_value in enumerate(recall_inds_list):
                            # Ensure pi is within the range of both arrays
                            pos_idx_int: int = int(pos_idx_value)
                            if 0 <= pos_idx_int < len(pr) and 0 <= pos_idx_int < len(
                                dt_scores_sorted
                            ):
                                precision_at_recall[ri] = pr[pos_idx_int]
                                score_at_recall[ri] = dt_scores_sorted[pos_idx_int]

                        # Convert precision to numpy array
                        precision[iou_thresh_idx, :, cat_idx, area_idx, max_det_idx] = (
                            np.array(precision_at_recall, dtype=np.float32)
                        )
                        # Convert scores to numpy array
                        scores[iou_thresh_idx, :, cat_idx, area_idx, max_det_idx] = (
                            np.array(score_at_recall, dtype=np.float32)
                        )

        self.results = {
            "params": self.params,
            "counts": [
                num_iou_thresholds,
                num_recall_thresholds,
                num_categories,
                num_area_ranges,
                num_max_detections,
            ],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "scores": scores,
        }

        # Helper function to compute average precision while handling -1 sentinel values
        def compute_average_precision(
            precision_slice: npt.NDArray[np.float32],
        ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
            """Compute average precision while handling -1 sentinel values."""
            valid_mask = precision_slice != -1
            valid_precision = np.where(valid_mask, precision_slice, np.float32(0.0))

            def mean_with_mask(
                axis: int | tuple[int, ...],
            ) -> npt.NDArray[np.float32]:
                sums = valid_precision.sum(axis=axis, dtype=np.float64)
                counts = valid_mask.sum(axis=axis)
                means = np.divide(
                    sums,
                    counts,
                    out=np.full(sums.shape, -1.0, dtype=np.float64),
                    where=counts > 0,
                )
                return means.astype(np.float32)

            mAP_scores = mean_with_mask((1, 2))
            ap_per_class = mean_with_mask(1).transpose(1, 0)
            return mAP_scores, ap_per_class

        # Average precision over all sizes, 100 max detections
        area_range_idx = list(ObjectSize).index(ObjectSize.ALL)
        max_100_dets_idx = self.params.max_dets.index(100)
        # Average precision  [threshold, recall, classes]
        average_precision_all_sizes = precision[
            :, :, :, area_range_idx, max_100_dets_idx
        ]
        # mAP over thresholds (dimension=num_thresholds)
        # Exclude -1 sentinel values when computing mean
        mAP_scores_all_sizes, ap_per_class_all_sizes = compute_average_precision(
            average_precision_all_sizes
        )

        # Average precision for SMALL objects and 100 max detections
        small_area_range_idx = list(ObjectSize).index(ObjectSize.SMALL)
        average_precision_small = precision[
            :, :, :, small_area_range_idx, max_100_dets_idx
        ]
        mAP_scores_small, ap_per_class_small = compute_average_precision(
            average_precision_small
        )

        # Average precision for MEDIUM objects and 100 max detections
        medium_area_range_idx = list(ObjectSize).index(ObjectSize.MEDIUM)
        average_precision_medium = precision[
            :, :, :, medium_area_range_idx, max_100_dets_idx
        ]
        mAP_scores_medium, ap_per_class_medium = compute_average_precision(
            average_precision_medium
        )

        # Average precision for LARGE objects and 100 max detections
        large_area_range_idx = list(ObjectSize).index(ObjectSize.LARGE)
        average_precision_large = precision[
            :, :, :, large_area_range_idx, max_100_dets_idx
        ]
        mAP_scores_large, ap_per_class_large = compute_average_precision(
            average_precision_large
        )

        self.results = {
            "params": self.params,
            "counts": [
                num_iou_thresholds,
                num_recall_thresholds,
                num_categories,
                num_area_ranges,
                num_max_detections,
            ],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "scores": scores,
            "mAP_scores_all_sizes": mAP_scores_all_sizes,
            "ap_per_class_all_sizes": ap_per_class_all_sizes,
            "mAP_scores_small": mAP_scores_small,
            "ap_per_class_small": ap_per_class_small,
            "mAP_scores_medium": mAP_scores_medium,
            "ap_per_class_medium": ap_per_class_medium,
            "mAP_scores_large": mAP_scores_large,
            "ap_per_class_large": ap_per_class_large,
        }

    def _pycocotools_summarize(self) -> None:
        """
        Compute and display summary metrics for evaluation results.
        """

        def _summarize(
            use_ap: bool = True,
            iou_thr: float | None = None,
            area_range: ObjectSize = ObjectSize.ALL,
            max_dets: int = 100,
        ) -> float:
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.10f}"
            titleStr = "Average Precision" if use_ap else "Average Recall"
            typeStr = "(AP)" if use_ap else "(AR)"
            iou_str = (
                f"{self.params.iou_thrs[0]:0.2f}:{self.params.iou_thrs[-1]:0.2f}"
                if iou_thr is None
                else f"{iou_thr:0.2f}"
            )
            all_object_sizes = list(ObjectSize)
            area_range_idx = all_object_sizes.index(area_range)
            max_detections_idx = self.params.max_dets.index(max_dets)
            if use_ap:
                # Dimension of precision:
                # threshold x recall x classes x areas x max detections
                s = self.results["precision"]
                # IOU
                if iou_thr is not None:
                    t = np.where(iou_thr == self.params.iou_thrs)[0]
                    s = s[t]
                s = s[:, :, :, area_range_idx, max_detections_idx]
            else:
                # Dimension of recall:
                # threshold x classes x areas x max detections
                s = self.results["recall"]
                if iou_thr is not None:
                    t = np.where(iou_thr == self.params.iou_thrs)[0]
                    s = s[t]
                s = s[:, :, area_range_idx, max_detections_idx]
            if len(s[s > -1]) == 0:
                mean_s = -1.0
            else:
                mean_s = float(np.mean(s[s > -1]))
            logger.info(
                iStr.format(titleStr, typeStr, iou_str, area_range, max_dets, mean_s)
            )
            return mean_s

        def _summarize_predictions() -> npt.NDArray[np.float32]:
            stats: npt.NDArray[np.float32] = np.zeros((12,), dtype=np.float32)
            stats[0] = _summarize(use_ap=True)
            stats[1] = _summarize(
                use_ap=True, iou_thr=0.5, max_dets=self.params.max_dets[2]
            )
            stats[2] = _summarize(
                use_ap=True, iou_thr=0.75, max_dets=self.params.max_dets[2]
            )
            stats[3] = _summarize(
                use_ap=True,
                area_range=ObjectSize.SMALL,
                max_dets=self.params.max_dets[2],
            )
            stats[4] = _summarize(
                use_ap=True,
                area_range=ObjectSize.MEDIUM,
                max_dets=self.params.max_dets[2],
            )
            stats[5] = _summarize(
                use_ap=True,
                area_range=ObjectSize.LARGE,
                max_dets=self.params.max_dets[2],
            )
            stats[6] = _summarize(use_ap=False, max_dets=self.params.max_dets[0])
            stats[7] = _summarize(use_ap=False, max_dets=self.params.max_dets[1])
            stats[8] = _summarize(use_ap=False, max_dets=self.params.max_dets[2])
            stats[9] = _summarize(
                use_ap=False,
                area_range=ObjectSize.SMALL,
                max_dets=self.params.max_dets[2],
            )
            stats[10] = _summarize(
                use_ap=False,
                area_range=ObjectSize.MEDIUM,
                max_dets=self.params.max_dets[2],
            )
            stats[11] = _summarize(
                use_ap=False,
                area_range=ObjectSize.LARGE,
                max_dets=self.params.max_dets[2],
            )
            return stats

        if len(self.results) != 0:
            self.stats = _summarize_predictions().tolist()

    def evaluate(self) -> None:
        """
        Start the per image evaluation on all images and keeep results in
        self.eval_imgs (a list of dictionaries).
        """
        # Select all parameters to evaluate
        self.params.img_ids = list(np.unique(self.params.img_ids))
        self.params.cat_ids = list(np.unique(self.params.cat_ids))
        self.params.max_dets = sorted(self.params.max_dets)

        self._prepare_targets_and_predictions()

        # Compute IOUs between all targets and predictions for all images and categories
        self.ious = {
            (img_id, cat_id): self._compute_iou(img_id, cat_id)
            for img_id in self.params.img_ids
            for cat_id in self.params.cat_ids
        }

        # Select the largest max area (the last element containing 100 dets
        max_det = self.params.max_dets[-1]

        # Evaluate each image with all categories, area range and max detections
        self.eval_imgs = [
            self._evaluate_image(img_id, cat_id, area_range, max_det)
            for cat_id in self.params.cat_ids
            for area_range in self.params.area_range
            for img_id in self.params.img_ids
        ]

        # Accumulate results
        self._accumulate()


class MeanAveragePrecision(Metric):
    """
    Mean Average Precision (mAP) is a metric used to evaluate object detection models.
    It is the average of the precision-recall curves at different IoU thresholds.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> from supervision.metrics import MeanAveragePrecision
        >>> predictions = sv.Detections(
        ...     xyxy=np.array([[0, 0, 10, 10]]),
        ...     class_id=np.array([0]),
        ...     confidence=np.array([0.9])
        ... )
        >>> targets = sv.Detections(
        ...     xyxy=np.array([[0, 0, 10, 10]]),
        ...     class_id=np.array([0])
        ... )
        >>> map_metric = MeanAveragePrecision()
        >>> map_result = map_metric.update(predictions, targets).compute()
        >>> round(float(map_result.map50), 2)
        1.0
        >>> print(map_result)
        Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 1.000
        Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 1.000
        Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 1.000
        Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 1.000
        Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
        Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000

        ```

    ![example_plot](
        https://media.roboflow.com/supervision-docs/metrics/mAP_plot_example.png
    ){ align=center width="800" }
    """

    def __init__(
        self,
        metric_target: MetricTarget = MetricTarget.BOXES,
        class_agnostic: bool = False,
        class_mapping: dict[int, int] | None = None,
        image_indices: list[int] | None = None,
    ):
        """
        Initialize the Mean Average Precision metric.

        Args:
            metric_target: The type of detection data to use.
            class_agnostic: Whether to treat all data as a single class.
            class_mapping: A dictionary to map class IDs to new IDs.
            image_indices: The indices of the images to use.
        """
        self._metric_target = metric_target
        self._class_agnostic = class_agnostic

        self._predictions_list: list[Detections] = []
        self._targets_list: list[Detections] = []
        self._class_mapping = class_mapping
        self._image_indices = image_indices

    def reset(self) -> None:
        """
        Reset the metric to its initial state, clearing all stored data.
        """
        self._predictions_list = []
        self._targets_list = []

    def update(
        self,
        predictions: Detections | list[Detections],
        targets: Detections | list[Detections],
    ) -> MeanAveragePrecision:
        """
        Add new predictions and targets to the metric, but do not compute the result.

        Args:
            predictions: The predicted detections.
            targets: The ground-truth detections.

        Returns:
            The updated metric instance.
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

        if self._class_agnostic:
            predictions = deepcopy(predictions)
            targets = deepcopy(targets)

            for prediction in predictions:
                if prediction.class_id is not None:
                    prediction.class_id[:] = -1
            for target in targets:
                if target.class_id is not None:
                    target.class_id[:] = -1

        self._predictions_list.extend(predictions)
        self._targets_list.extend(targets)

        return self

    def _prepare_targets(
        self, targets: list[Detections]
    ) -> dict[str, list[dict[str, Any]]]:
        """Transform targets into a dictionary that can be used by the COCO evaluator"""
        images = [{"id": img_id} for img_id in range(len(targets))]
        if self._image_indices is not None:
            images = [{"id": self._image_indices[img["id"]]} for img in images]
        # Annotations list
        annotations: list[dict[str, Any]] = []
        for image_id, image_targets in enumerate(targets):
            if self._image_indices is not None:
                image_id = self._image_indices[image_id]

            # Ensure xyxy is not None
            if image_targets.xyxy is None:
                continue

            for target_idx, xyxy in enumerate(image_targets.xyxy):
                xywh = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]

                # Default values
                category_id = 0

                if image_targets.class_id is not None:
                    cls_id = image_targets.class_id[target_idx]
                    if self._class_mapping is not None:
                        category_id = self._class_mapping[int(cls_id)]
                    else:
                        category_id = int(cls_id)

                # Use area from data if available, otherwise calculate from bbox
                area = None
                if image_targets.data is not None and "area" in image_targets.data:
                    area = float(image_targets.data["area"][target_idx])

                if area is None:
                    area = xywh[2] * xywh[3]

                iscrowd = 0
                if image_targets.data is not None and "iscrowd" in image_targets.data:
                    iscrowd = int(image_targets.data["iscrowd"][target_idx])

                dict_annotation = {
                    "area": area,
                    "iscrowd": iscrowd,
                    "image_id": image_id,
                    "bbox": xywh,
                    "category_id": category_id,
                    "id": len(annotations) + 1,  # Start IDs from 1 (0 means no match)
                    "ignore": 0,
                }
                annotations.append(dict_annotation)
        # Category list
        all_cat_ids = {annotation.get("category_id") for annotation in annotations}
        categories = [{"id": cat_id} for cat_id in all_cat_ids]
        # Create coco dictionary
        return {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

    def _prepare_predictions(
        self, predictions: list[Detections]
    ) -> list[dict[str, Any]]:
        """Transform predictions into a list of predictions that can be used by the COCO
        evaluator."""
        coco_predictions: list[dict[str, Any]] = []
        for image_id, image_predictions in enumerate(predictions):
            if self._image_indices is not None:
                image_id = self._image_indices[image_id]

            if image_predictions.xyxy is None:
                continue

            for pred_idx, xyxy in enumerate(image_predictions.xyxy):
                xywh = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]

                category_id = 0
                score = 0.0

                if image_predictions.class_id is not None:
                    cls_id = image_predictions.class_id[pred_idx]
                    if self._class_mapping is not None:
                        category_id = self._class_mapping[int(cls_id)]
                    else:
                        category_id = int(cls_id)

                if image_predictions.confidence is not None:
                    score = float(image_predictions.confidence[pred_idx])

                # Use area from data if available, otherwise calculate from bbox
                area = None
                if (
                    image_predictions.data is not None
                    and "area" in image_predictions.data
                ):
                    area = float(image_predictions.data["area"][pred_idx])

                if area is None:
                    area = xywh[2] * xywh[3]

                dict_prediction = {
                    "image_id": image_id,
                    "bbox": xywh,
                    "score": score,
                    "category_id": category_id,
                    "area": area,
                    "id": len(coco_predictions) + 1,
                }
                coco_predictions.append(dict_prediction)
        return coco_predictions

    def compute(self) -> MeanAveragePrecisionResult:
        """
        Calculate Mean Average Precision based on predicted and ground-truth
        detections at different thresholds using the COCO evaluation metrics.
        Source: https://github.com/rafaelpadilla/review_object_detection_metrics

        Returns:
            The Mean Average Precision result.
        """
        total_images_predictions = len(self._predictions_list)
        total_images_targets = len(self._targets_list)

        if total_images_predictions != total_images_targets:
            raise ValueError(
                f"The number of predictions ({total_images_predictions}) and"
                f" targets ({total_images_targets}) during the evaluation must be"
                " the same."
            )
        dict_targets = self._prepare_targets(self._targets_list)
        lst_predictions = self._prepare_predictions(self._predictions_list)
        # Create a coco object with the targets
        coco_gt = EvaluationDataset(targets=dict_targets)
        # Include the predictions to coco object
        coco_det = coco_gt.load_predictions(lst_predictions)
        # Create a coco evaluator with the predictions
        cocoEval = COCOEvaluator(coco_gt, coco_det)

        # Evaluate on all images
        cocoEval.evaluate()

        # Create MeanAveragePrecisionResult object for small objects
        mAP_small = MeanAveragePrecisionResult(
            metric_target=self._metric_target,
            is_class_agnostic=self._class_agnostic,
            mAP_scores=cocoEval.results["mAP_scores_small"],
            ap_per_class=cocoEval.results["ap_per_class_small"],
            iou_thresholds=cocoEval.params.iou_thrs,
            matched_classes=np.array(cocoEval.params.cat_ids),
        )
        # Create MeanAveragePrecisionResult object for medium objects
        mAP_medium = MeanAveragePrecisionResult(
            metric_target=self._metric_target,
            is_class_agnostic=self._class_agnostic,
            mAP_scores=cocoEval.results["mAP_scores_medium"],
            ap_per_class=cocoEval.results["ap_per_class_medium"],
            iou_thresholds=cocoEval.params.iou_thrs,
            matched_classes=np.array(cocoEval.params.cat_ids),
        )
        # Create MeanAveragePrecisionResult object for large objects
        mAP_large = MeanAveragePrecisionResult(
            metric_target=self._metric_target,
            is_class_agnostic=self._class_agnostic,
            mAP_scores=cocoEval.results["mAP_scores_large"],
            ap_per_class=cocoEval.results["ap_per_class_large"],
            iou_thresholds=cocoEval.params.iou_thrs,
            matched_classes=np.array(cocoEval.params.cat_ids),
        )

        # Create the final MeanAveragePrecisionResult object
        mAP_result = MeanAveragePrecisionResult(
            metric_target=self._metric_target,
            is_class_agnostic=self._class_agnostic,
            mAP_scores=cocoEval.results["mAP_scores_all_sizes"],
            ap_per_class=cocoEval.results["ap_per_class_all_sizes"],
            iou_thresholds=cocoEval.params.iou_thrs,
            matched_classes=np.array(cocoEval.params.cat_ids),
            small_objects=mAP_small,
            medium_objects=mAP_medium,
            large_objects=mAP_large,
        )
        return mAP_result
