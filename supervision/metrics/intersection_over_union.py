from typing import Dict, List, Union

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from supervision.detection.core import Detections
from supervision.detection.utils import box_iou_batch
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
        data_1: Union[Detections, List[Detections]],
        data_2: Union[Detections, List[Detections]],
    ) -> Self:
        """
        Add data to the metric, without computing the result.

        Args:
            data_1 (Union[Detection, List[Detections]]): The first set of data.
            data_2 (Union[Detection, List[Detections]]): The second set of data.

        Returns:
            Metric: The metric object itself. You can get the metric result
            by calling the `compute` method.
        """

        if isinstance(data_1, list):
            for d1 in data_1:
                self.update(d1, Detections.empty())
        else:
            self._update(data_1, Detections.empty())

        if isinstance(data_2, list):
            for d2 in data_2:
                self.update(Detections.empty(), d2)
        else:
            self._update(Detections.empty(), data_2)

        return self

    def _update(
        self,
        data_1: Union[Detections],
        data_2: Union[Detections],
    ) -> Self:
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
        ious = {}
        for class_id, array_1, array_2 in self._store:
            if self._metric_target == MetricTarget.BOXES:
                if array_1 is None or array_2 is None:
                    ious[class_id] = np.empty((0, 4), dtype=np.float32)
                    continue
                iou = box_iou_batch(array_1, array_2)

            else:
                raise NotImplementedError(
                    "Intersection over union is not implemented"
                    f" for {self._metric_target}."
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
