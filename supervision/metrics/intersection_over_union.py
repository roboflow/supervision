from typing import Dict, Union

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from supervision.detection.core import Detections
from supervision.metrics.core import InternalMetricDataStore, Metric, MetricTarget


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
    ) -> Self:
        """
        Add data to the metric, without computing the result.

        The arguments can be:

        * Boxes of shape `(N, 4)`, `float32`,
        * Masks of shape `(N, H, W)`, `bool`
        * Oriented bounding boxes of shape `(N, 8)`, `float32`.
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
                    "Intersection over union is not implemented"
                    " for {self._metric_target}."
                )
            ious[class_id] = iou
        return ious

    # TODO: This would return dict[int, pd.DataFrame]. Do we want that?
    #       It'd be cleaner if it returned a single DataFrame, but the sizes
    #       differ if class_agnostic=False.

    # def to_pandas(self) -> 'pd.DataFrame':
    #     """
    #     Return a pandas DataFrame representation of the metric.
    #     """
    #     self._ensure_pandas_installed()
    #     import pandas as pd

    #     # TODO: use cached results
    #     ious = self.compute()
    #     print(len(ious))

    #     class_names = []
    #     arrays = []

    #     for class_id, array in ious.items():
    #         print(array.shape)
    #         class_names.append(np.full(array.shape[0], class_id))
    #         arrays.append(array)
    #     stacked_class_ids = np.concatenate(class_names)
    #     stacked_ious = np.vstack(arrays)
    #     combined = np.column_stack((stacked_class_ids, stacked_ious))

    #     column_names = \
    # ['class_id'] + [f'col_{i+1}' for i in range(stacked_ious.shape[1])]
    #     result = pd.DataFrame(combined, columns=column_names)

    #     return result

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
