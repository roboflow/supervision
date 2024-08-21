from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Iterator, Set, Tuple

import numpy as np
import numpy.typing as npt

from supervision import config
from supervision.detection.core import Detections
from supervision.metrics.utils import pad_mask

CLASS_ID_NONE = -1
CONFIDENCE_NONE = -1
"""Used by metrics module as class ID, when none is present"""


class Metric(ABC):
    """
    The base class for all supervision metrics.
    """

    @abstractmethod
    def update(self, *args, **kwargs) -> "Metric":
        """
        Add data to the metric, without computing the result.
        Return the metric itself to allow method chaining.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Reset internal metric state.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self, *args, **kwargs) -> Any:
        """
        Compute the metric from the internal state and return the result.
        """
        raise NotImplementedError

    def _ensure_pandas_installed(self):
        try:
            import pandas  # noqa
        except ImportError:
            raise ImportError(
                "Function `to_pandas` requires the `metrics` extra to be installed."
                " Run `pip install 'supervision[metrics]'` or"
                " `poetry add supervision -E metrics`."
            )


class MetricTarget(Enum):
    """
    Specifies what type of detection is used to compute the metric.

    * BOXES: xyxy bounding boxes
    * MASKS: Binary masks
    * ORIENTED_BOUNDING_BOXES: Oriented bounding boxes (OBB)
    """

    BOXES = "boxes"
    MASKS = "masks"
    ORIENTED_BOUNDING_BOXES = "obb"


class UnsupportedMetricTargetError(Exception):
    """
    Raised when a metric does not support the specified target. (and never will!)
    If the support might be added in the future, raise `NotImplementedError` instead.
    """

    def __init__(self, metric: Metric, target: MetricTarget):
        super().__init__(f"Metric {metric} does not support target {target}")


class MetricData:
    """
    A container for detection contents, decouple from Detections.
    While a np.ndarray work for xyxy and obb, this approach solves
    the mask concatenation problem.
    """

    def __init__(self, metric_target: MetricTarget, class_agnostic: bool = False):
        self._metric_target = metric_target
        self._class_agnostic = class_agnostic
        self.confidence = np.array([], dtype=np.float32)
        self.class_id = np.array([], dtype=int)
        self.data: npt.NDArray = self._get_empty_data()

    def update(self, detections: Detections):
        """Add new detections to the store."""
        new_data = self._get_content(detections)
        self._validate_shape(new_data)

        if self._metric_target == MetricTarget.BOXES:
            self._append_boxes(new_data)
        elif self._metric_target == MetricTarget.MASKS:
            self._append_mask(new_data)
        elif self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
            self.data = np.vstack((self.data, new_data))

        confidence = self._get_confidence(detections)
        self._append_confidence(confidence)

        class_id = self._get_class_id(detections)
        self._append_class_id(class_id)

        if len(self.class_id) != len(self.confidence) or len(self.class_id) != len(
            self.data
        ):
            raise ValueError(
                f"Inconsistent data length: class_id={len(class_id)},"
                f" confidence={len(confidence)}, data={len(new_data)}"
            )

    def get_classes(self) -> Set[int]:
        """Return all class IDs."""
        return set(self.class_id)

    def get_subset_by_class(self, class_id: int) -> MetricData:
        """Return data, confidence and class_id for a specific class."""
        mask = self.class_id == class_id
        new_data_obj = MetricData(self._metric_target)
        new_data_obj.data = self.data[mask]
        new_data_obj.confidence = self.confidence[mask]
        new_data_obj.class_id = self.class_id[mask]
        return new_data_obj

    def __len__(self) -> int:
        return len(self.data)

    def _get_content(self, detections: Detections) -> npt.NDArray:
        """Return boxes, masks or oriented bounding boxes from the data."""
        if self._metric_target == MetricTarget.BOXES:
            return detections.xyxy
        if self._metric_target == MetricTarget.MASKS:
            return (
                detections.mask
                if detections.mask is not None
                else self._get_empty_data()
            )
        if self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
            obb = detections.data.get(
                config.ORIENTED_BOX_COORDINATES, self._get_empty_data()
            )
            return np.ndarray(obb, dtype=np.float32)
        raise ValueError(f"Invalid metric target: {self._metric_target}")

    def _get_class_id(self, detections: Detections) -> npt.NDArray[np.int_]:
        if self._class_agnostic or detections.class_id is None:
            return np.array([CLASS_ID_NONE] * len(detections), dtype=int)
        return detections.class_id

    def _get_confidence(self, detections: Detections) -> npt.NDArray[np.float32]:
        if detections.confidence is None:
            return np.full(len(detections), -1, dtype=np.float32)
        return detections.confidence

    def _append_class_id(self, new_class_id: npt.NDArray[np.int_]) -> None:
        self.class_id = np.hstack((self.class_id, new_class_id))

    def _append_confidence(self, new_confidence: npt.NDArray[np.float32]) -> None:
        self.confidence = np.hstack((self.confidence, new_confidence))

    def _append_boxes(self, new_boxes: npt.NDArray[np.float32]) -> None:
        """Stack new xyxy or obb boxes on top of stored boxes."""
        if self._metric_target not in [
            MetricTarget.BOXES,
            MetricTarget.ORIENTED_BOUNDING_BOXES,
        ]:
            raise ValueError("This method is only for box data.")
        self.data = np.vstack((self.data, new_boxes))

    def _append_mask(self, new_mask: npt.NDArray[np.bool_]) -> None:
        """Stack new mask onto stored masks. Expand the shapes if necessary."""
        if self._metric_target != MetricTarget.MASKS:
            raise ValueError("This method is only for mask data.")
        self._validate_shape(new_mask)

        new_width = max(self.data.shape[1], new_mask.shape[1])
        new_height = max(self.data.shape[2], new_mask.shape[2])

        data = pad_mask(self.data, (new_width, new_height))
        new_mask = pad_mask(new_mask, (new_width, new_height))

        self.data = np.vstack((data, new_mask))

    def _get_empty_data(self) -> npt.NDArray:
        if self._metric_target == MetricTarget.BOXES:
            return np.empty((0, 4), dtype=np.float32)
        if self._metric_target == MetricTarget.MASKS:
            return np.empty((0, 0, 0), dtype=bool)
        if self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
            return np.empty((0, 8), dtype=np.float32)
        raise ValueError(f"Invalid metric target: {self._metric_target}")

    def _validate_shape(self, data: npt.NDArray) -> None:
        if self._metric_target == MetricTarget.BOXES:
            if len(data.shape) != 2 or data.shape[1] != 4:
                raise ValueError(f"Invalid xyxy shape: {data.shape}. Expected: (N, 4)")
        elif self._metric_target == MetricTarget.MASKS:
            if len(data.shape) != 3:
                raise ValueError(
                    f"Invalid mask shape: {data.shape}. Expected: (N, H, W)"
                )
        elif self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
            if len(data.shape) != 2 or data.shape[1] != 8:
                raise ValueError(f"Invalid obb shape: {data.shape}. Expected: (N, 8)")
        else:
            raise ValueError(f"Invalid metric target: {self._metric_target}")


class InternalMetricDataStore:
    """
    Stores internal data for metrics.

    Provides a class-agnostic way to access it.
    """

    def __init__(self, metric_target: MetricTarget, class_agnostic: bool = False):
        self._metric_target = metric_target
        self._class_agnostic = class_agnostic
        self._data_1: MetricData
        self._data_2: MetricData
        self.reset()

    def reset(self) -> None:
        self._data_1 = MetricData(self._metric_target, self._class_agnostic)
        self._data_2 = MetricData(self._metric_target, self._class_agnostic)

    def update(self, data_1: Detections, data_2: Detections) -> None:
        """
        Add new data to the store.

        Use sv.Detections.empty() if only one set of data is available.
        """
        self._data_1.update(data_1)
        self._data_2.update(data_2)

    def __getitem__(self, class_id: int) -> Tuple[MetricData, MetricData]:
        return (
            self._data_1.get_subset_by_class(class_id),
            self._data_2.get_subset_by_class(class_id),
        )

    def __iter__(self) -> Iterator[Tuple[int, MetricData, MetricData]]:
        all_classes = set.union(
            self._data_1.get_classes(),
            self._data_2.get_classes(),
        )
        for class_id in all_classes:
            yield class_id, *self[class_id]
