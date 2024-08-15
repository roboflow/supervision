from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Iterator, Tuple, Union

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from supervision import config
from supervision.detection.core import Detections
from supervision.metrics.utils import len0_like, pad_mask

CLASS_ID_NONE = -1
"""Used by metrics module as class ID, when none is present"""


class Metric(ABC):
    """
    The base class for all supervision metrics.
    """

    @abstractmethod
    def update(self, *args, **kwargs) -> Self:
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


class InternalMetricDataStore:
    """
    Stores internal data of IntersectionOverUnion metric:
    * Stores the basic data: boxes, masks, or oriented bounding boxes
    * Validates data: ensures data types and shape are consistent
    * Provides iteration by class

    Provides a class-agnostic mode, where all data is treated as a single class.
    Warning: numpy inputs are always considered as class-agnostic data.

    Data here refers to content of Detections objects: boxes, masks,
    or oriented bounding boxes.
    """

    def __init__(self, metric_target: MetricTarget, class_agnostic: bool):
        self._metric_target = metric_target
        self._class_agnostic = class_agnostic
        self._data_1: Dict[int, npt.NDArray]
        self._data_2: Dict[int, npt.NDArray]
        self._mask_shape: Tuple[int, int]
        self.reset()

    def reset(self) -> None:
        self._data_1 = {}
        self._data_2 = {}
        self._mask_shape = (0, 0)

    def update(
        self,
        data_1: Union[npt.NDArray, Detections],
        data_2: Union[npt.NDArray, Detections],
    ) -> None:
        """
        Add new data to the store.

        Use sv.Detections.empty() if only one set of data is available.
        """
        content_1 = self._get_content(data_1)
        content_2 = self._get_content(data_2)
        self._validate_shape(content_1)
        self._validate_shape(content_2)

        class_ids_1 = self._get_class_ids(data_1)
        class_ids_2 = self._get_class_ids(data_2)
        self._validate_class_ids(class_ids_1, class_ids_2)

        if self._metric_target == MetricTarget.MASKS:
            content_1 = self._expand_mask_shape(content_1)
            content_2 = self._expand_mask_shape(content_2)

        for class_id in set(class_ids_1):
            content_of_class = content_1[class_ids_1 == class_id]
            stored_content_of_class = self._data_1.get(class_id, len0_like(content_1))
            self._data_1[class_id] = np.vstack(
                (stored_content_of_class, content_of_class)
            )

        for class_id in set(class_ids_2):
            content_of_class = content_2[class_ids_2 == class_id]
            stored_content_of_class = self._data_2.get(class_id, len0_like(content_2))
            self._data_2[class_id] = np.vstack(
                (stored_content_of_class, content_of_class)
            )

    def __getitem__(self, class_id: int) -> Tuple[npt.NDArray, npt.NDArray]:
        return (
            self._data_1.get(class_id, self._make_empty()),
            self._data_2.get(class_id, self._make_empty()),
        )

    def __iter__(
        self,
    ) -> Iterator[Tuple[int, npt.NDArray, npt.NDArray]]:
        class_ids = sorted(set(self._data_1.keys()) | set(self._data_2.keys()))
        for class_id in class_ids:
            yield (
                class_id,
                *self[class_id],
            )

    def _get_content(self, data: Union[npt.NDArray, Detections]) -> npt.NDArray:
        """Return boxes, masks or oriented bounding boxes from the data."""
        if not isinstance(data, (Detections, np.ndarray)):
            raise ValueError(
                f"Invalid data type: {type(data)}."
                f" Only Detections or np.ndarray are supported."
            )
        if isinstance(data, np.ndarray):
            return data

        if self._metric_target == MetricTarget.BOXES:
            return data.xyxy
        if self._metric_target == MetricTarget.MASKS:
            return (
                data.mask if data.mask is not None else np.zeros((0, 0, 0), dtype=bool)
            )
        if self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
            obb = data.data.get(
                config.ORIENTED_BOX_COORDINATES, np.zeros((0, 8), dtype=np.float32)
            )
            return np.array(obb, dtype=np.float32)
        raise ValueError(f"Invalid metric target: {self._metric_target}")

    def _get_class_ids(
        self, data: Union[npt.NDArray, Detections]
    ) -> npt.NDArray[np.int_]:
        """
        Return an array of class IDs from the data. Guaranteed to
        match the length of data.
        """
        if (
            self._class_agnostic
            or isinstance(data, np.ndarray)
            or data.class_id is None
        ):
            return np.array([CLASS_ID_NONE] * len(data), dtype=int)
        return data.class_id

    def _validate_class_ids(
        self, class_id_1: npt.NDArray[np.int_], class_id_2: npt.NDArray[np.int_]
    ) -> None:
        class_set = set(class_id_1) | set(class_id_2)
        if len(class_set) >= 2 and CLASS_ID_NONE in class_set:
            raise ValueError(
                "Metrics cannot mix data with class ID and data without class ID."
            )

    def _validate_shape(self, data: npt.NDArray) -> None:
        shape = data.shape
        if self._metric_target == MetricTarget.BOXES:
            if len(shape) != 2 or shape[1] != 4:
                raise ValueError(f"Invalid xyxy shape: {shape}. Expected: (N, 4)")
        elif self._metric_target == MetricTarget.MASKS:
            if len(shape) != 3:
                raise ValueError(f"Invalid mask shape: {shape}. Expected: (N, H, W)")
        elif self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
            if len(shape) != 2 or shape[1] != 8:
                raise ValueError(f"Invalid obb shape: {shape}. Expected: (N, 8)")
        else:
            raise ValueError(f"Invalid metric target: {self._metric_target}")

    def _expand_mask_shape(self, data: npt.NDArray) -> npt.NDArray:
        """Pad the stored and new data to the same shape."""
        if self._metric_target != MetricTarget.MASKS:
            return data

        new_width = max(self._mask_shape[0], data.shape[1])
        new_height = max(self._mask_shape[1], data.shape[2])
        self._mask_shape = (new_width, new_height)

        data = pad_mask(data, self._mask_shape)

        for class_id, prev_data in self._data_1.items():
            self._data_1[class_id] = pad_mask(prev_data, self._mask_shape)
        for class_id, prev_data in self._data_2.items():
            self._data_2[class_id] = pad_mask(prev_data, self._mask_shape)

        return data

    def _make_empty(self) -> npt.NDArray:
        """Create an empty data object with the best-known shape for the target."""
        if self._metric_target == MetricTarget.BOXES:
            return np.empty((0, 4), dtype=np.float32)
        if self._metric_target == MetricTarget.MASKS:
            return np.empty((0, *self._mask_shape), dtype=bool)
        if self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
            return np.empty((0, 8), dtype=np.float32)
        raise ValueError(f"Invalid metric target: {self._metric_target}")
