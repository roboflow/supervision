from typing import TYPE_CHECKING, Dict, Iterator, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

import supervision.config as config
from supervision.detection.core import Detections
from supervision.metrics.core import Metric, MetricTarget

if TYPE_CHECKING:
    import pandas as pd


CLASS_ID_NONE = -1

Data = Union[npt.NDArray, Detections]


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
        self._datapoint_shape: Optional[Tuple[int]]
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

    # def update(
    #     self,
    #     data_1: Union[npt.NDArray, Detections],
    #     data_2: Union[npt.NDArray, Detections],
    # ) -> None:
    #     # This class dispatches to helper self._add_to_list, which does more validation
    #     # Then calls self._vstack which generates the final result
    #     if len(data_1) == 0 and len(data_2) == 0:
    #         return

    #     if type(data_1) != type(data_2):
    #         raise ValueError(
    #             f"Data types must match. Got {type(data_1)=} and {type(data_2)=}."
    #         )

    #     if isinstance(data_1, npt.NDArray):
    #         assert isinstance(data_2, npt.NDArray)
    #         self._update(data_1, class_id=CLASS_ID_NONE, data_id=1)
    #         self._update(data_2, class_id=CLASS_ID_NONE, data_id=2)
    #         return
    #     assert isinstance(data_1, Detections)
    #     assert isinstance(data_2, Detections)

    #     if self._class_agnostic:
    #         self._update(self._get_detections_content(data_1), class_id=CLASS_ID_NONE, data_id=1)
    #         self._update(self._get_detections_content(data_2), class_id=CLASS_ID_NONE, data_id=2)
    #         return

    #     if data_1.class_id is None:
    #         self._update(self._get_detections_content(data_1), class_id=CLASS_ID_NONE, data_id=1)
    #     else:
    #         for class_id in set(data_1.class_id):
    #             data_1_of_class = data_1[data_1.class_id == class_id]
    #             assert isinstance(data_1_of_class, Detections)
    #             self._update(self._get_detections_content(data_1_of_class), class_id=class_id, data_id=1)

    #     if data_2.class_id is None:
    #         self._update(self._get_detections_content(data_2), class_id=CLASS_ID_NONE, data_id=2)
    #     else:
    #         for class_id in set(data_2.class_id):
    #             data_2_of_class = data_2[data_2.class_id == class_id]
    #             assert isinstance(data_2_of_class, Detections)
    #             self._update(self._get_detections_content(data_2_of_class), class_id=class_id, data_id=2)

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

    def _get_content(self, data: Data) -> Optional[npt.NDArray]:
        """Return boxes, masks or oriented bounding boxes from the data."""
        if isinstance(data, npt.NDArray):
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

    def _get_class_ids(self, data: Data) -> npt.NDArray[np.int_]:
        if self._class_agnostic or isinstance(data, npt.NDArray):
            return np.array([CLASS_ID_NONE] * len(data), dtype=int)
        assert isinstance(data, Detections)
        if data.class_id is None:
            return np.array([CLASS_ID_NONE] * len(data), dtype=int)
        return data.class_id

    # def _get_detections_content(self, data: Detections) -> Optional[npt.NDArray]:
    #     if self._metric_target == MetricTarget.BOXES:
    #         return data.xyxy
    #     if self._metric_target == MetricTarget.MASKS:
    #         return data.mask
    #     if self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
    #         obb = data.data.get(config.ORIENTED_BOX_COORDINATES, None)
    #         if isinstance(obb, list):
    #             obb = np.array(obb, dtype=np.float32)
    #         return obb
    #     raise ValueError(f"Invalid metric target: {self._metric_target}")

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


class IntersectionOverUnion(Metric):
    def __init__(
        self,
        metric_target: MetricTarget = MetricTarget.BOXES,
        class_agnostic: bool = False,
        iou_threshold: float = 0.25,
    ):
        # TODO: implement for masks and oriented bounding boxes
        if metric_target in [MetricTarget.MASKS, MetricTarget.ORIENTED_BOUNDING_BOXES]:
            raise NotImplementedError(
                f"Intersection over union is not implemented for {metric_target}."
            )

        self._metric_target = metric_target
        self._class_agnostic = class_agnostic
        self._iou_threshold = iou_threshold

        self._store = InternalMetricDataStore(metric_target, class_agnostic)

    def reset(self) -> None:
        self._store.reset()

    def update(
        self,
        data_1: Union[npt.NDArray, Detections],
        data_2: Union[npt.NDArray, Detections],
    ) -> Metric:
        """
        Add data to the metric, without computing the result.

        The arguments can be:
        * Boxes of shape (N, 4), float32,
        * Masks of shape (N, H, W), bool
        * Oriented bounding boxes of shape (N, 8), float32.
        * Detections object.

        Args:
            data_1 (Union[npt.NDArray, Detection]): The first set of data.
            data_2 (Union[npt.NDArray, Detection]): The second set of data.

        Returns:
            Metric: The metric object itself. You can get the metric result
            by calling the `compute` method.
        """
        self._store.update(data_1, data_2)
        return self

    def compute(self) -> Dict[int, npt.NDArray[np.float32]]:
        """
        Compute the Intersection over Union metric (IoU)
        Uses the data set with the `update` method.

        Returns:
            Dict[int, npt.NDArray[np.float32]]: A dictionary with class IDs as keys.
            If no class ID is provided, the key is the value CLASS_ID_NONE.
        """
        # TODO: cache computed result.
        ious = {}
        for class_id, array_1, array_2 in self._store:
            if self._metric_target == MetricTarget.BOXES:
                if array_1 is None or array_2 is None:
                    ious[class_id] = np.empty((0, 4), dtype=np.float32)
                    continue
                iou = self._compute_box_iou(array_1, array_2)

            else:
                raise NotImplementedError(
                    f"Intersection over union is not implemented for {self._metric_target}."
                )
            ious[class_id] = iou
        return ious

    def to_pandas(self) -> pd.DataFrame:
        """
        Return a pandas DataFrame representation of the metric.
        """
        self._ensure_pandas_installed()
        import pandas as pd

        # TODO: use cache results instead
        ious = self.compute()

        # TODO: continue

        # data_frame = pd.DataFrame()

        # return s
        return pd.DataFrame()

    @staticmethod
    def _compute_box_iou(
        array_1: npt.NDArray, array_2: npt.NDArray
    ) -> npt.NDArray[np.float32]:
        """Computes the pairwise intersection-over-union between two sets of boxes."""

        def box_area(box):
            return (box[2] - box[0]) * (box[3] - box[1])

        area_true = box_area(array_1.T)
        area_detection = box_area(array_2.T)

        top_left = np.maximum(array_1[:, None, :2], array_2[:, :2])
        bottom_right = np.minimum(array_1[:, None, 2:], array_2[:, 2:])

        area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
        ious = area_inter / (area_true[:, None] + area_detection - area_inter)
        ious = np.nan_to_num(ious)
        return ious
