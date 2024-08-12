from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from supervision import config
from supervision.detection.core import Detections

"""Used by metrics module as class ID, when none is present"""
CLASS_ID_NONE = -1


class Metric(ABC):
    """
    The base class for all supervision metrics.
    """

    @abstractmethod
    def update(self, *args, **kwargs) -> Metric:
        """
        Add data to the metric, without computing the result.
        Return the metric itself to allow method chaining, for example:

        Example:
            ```python
            result = metric.update(data).compute()
            ```
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

    # TODO: determine if this is necessary.
    # @abstractmethod
    # def to_pandas(self, *args, **kwargs) -> Any:
    #     """
    #     Return a pandas DataFrame representation of the metric.
    #     """
    #     self._ensure_pandas_installed()
    #     raise NotImplementedError

    def _ensure_pandas_installed(self):
        try:
            import pandas
        except ImportError:
            raise ImportError(
                "Function `to_pandas` requires the `metrics` extra to be installed."
                " Run `pip install 'supervision[metrics]'` or `poetry add supervision -E metrics`."
            )


class MetricTarget(Enum):
    """
    Specifies what type of detection is used to compute the metric.
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
    """

    def __init__(self, metric_target: MetricTarget, class_agnostic: bool):
        self._metric_target = metric_target
        self._class_agnostic = class_agnostic
        self._data_1: Dict[int, npt.NDArray]
        self._data_2: Dict[int, npt.NDArray]
        self._datapoint_shape: Optional[Tuple[int, ...]]
        self.reset()

    def reset(self) -> None:
        self._data_1 = {}
        self._data_2 = {}
        if self._metric_target == MetricTarget.BOXES:
            self._datapoint_shape = (4,)
        elif self._metric_target == MetricTarget.MASKS:
            # Determined when adding data
            self._datapoint_shape = None
        elif self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
            self._datapoint_shape = (8,)

    def update(
        self,
        data_1: Union[npt.NDArray, Detections],
        data_2: Union[npt.NDArray, Detections],
    ) -> None:
        content_1 = self._get_content(data_1)
        content_2 = self._get_content(data_2)
        class_ids_1 = self._get_class_ids(data_1)
        class_ids_2 = self._get_class_ids(data_2)
        self._validate_class_ids(class_ids_1)
        self._validate_class_ids(class_ids_2)
        if content_1 is not None and len(content_1) > 0:
            assert len(content_1) == len(class_ids_1)
            for class_id in set(class_ids_1):
                content_of_class = content_1[class_ids_1 == class_id]
                if class_id not in self._data_1:
                    self._data_1[class_id] = content_of_class
                    continue
                self._data_1[class_id] = np.vstack(
                    (self._data_1[class_id], content_of_class)
                )

        if content_2 is not None and len(content_2) > 0:
            assert len(content_2) == len(class_ids_2)
            for class_id in set(class_ids_2):
                content_of_class = content_2[class_ids_2 == class_id]
                if class_id not in self._data_2:
                    self._data_2[class_id] = content_of_class
                    continue
                self._data_2[class_id] = np.vstack(
                    (self._data_2[class_id], content_of_class)
                )

    def __iter__(
        self,
    ) -> Iterator[Tuple[int, Optional[npt.NDArray], Optional[npt.NDArray]]]:
        class_ids = sorted(
            set.union(set(self._data_1.keys()), set(self._data_2.keys()))
        )
        for class_id in class_ids:
            yield (
                class_id,
                self._data_1.get(class_id, None),
                self._data_2.get(class_id, None),
            )

    def _get_content(
        self, data: Union[npt.NDArray, Detections]
    ) -> Optional[npt.NDArray]:
        """Return boxes, masks or oriented bounding boxes from the data."""
        if isinstance(data, np.ndarray):
            return data
        assert isinstance(data, Detections)

        if self._metric_target == MetricTarget.BOXES:
            return data.xyxy
        if self._metric_target == MetricTarget.MASKS:
            return data.mask
        if self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
            obb = data.data.get(config.ORIENTED_BOX_COORDINATES, None)
            if isinstance(obb, list):
                obb = np.array(obb, dtype=np.float32)
            return obb
        raise ValueError(f"Invalid metric target: {self._metric_target}")

    def _get_class_ids(
        self, data: Union[npt.NDArray, Detections]
    ) -> npt.NDArray[np.int_]:
        if self._class_agnostic or isinstance(data, np.ndarray):
            return np.array([CLASS_ID_NONE] * len(data), dtype=int)
        assert isinstance(data, Detections)
        if data.class_id is None:
            return np.array([CLASS_ID_NONE] * len(data), dtype=int)
        return data.class_id

    def _validate_class_ids(self, class_id: npt.NDArray[np.int_]) -> None:
        class_set = set(class_id)
        if len(class_set) >= 2 and -1 in class_set:
            raise ValueError(
                "Metrics store received results with partially defined classes."
            )

    def _validate_shape(self, data: npt.NDArray) -> None:
        if self._datapoint_shape is None:
            assert self._metric_target == MetricTarget.MASKS
            self._datapoint_shape = data.shape[1:]
            return
        if data.shape[1:] != self._datapoint_shape:
            raise ValueError(
                f"Invalid data shape: {data.shape}. Expected: (N, {self._datapoint_shape})"
            )
