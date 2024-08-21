from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import matplotlib.pyplot as plt  # noqa
import numpy as np
import numpy.typing as npt
from matplotlib.colors import LinearSegmentedColormap

from supervision.detection.core import Detections
from supervision.detection.utils import box_iou_batch, mask_iou_batch
from supervision.metrics.core import Metric, MetricTarget
from supervision.metrics.utils.internal_data_store import MetricDataStore
from supervision.metrics.utils.utils import ensure_pandas_installed

if TYPE_CHECKING:
    import pandas as pd


class IntersectionOverUnion(Metric):
    def __init__(
        self,
        metric_target: MetricTarget = MetricTarget.BOXES,
        class_agnostic: bool = False,
        shared_data_store: Optional[MetricDataStore] = None,
    ):
        """
        Initialize the Intersection over Union metric.

        Args:
            metric_target (MetricTarget): The type of detection data to use.
            class_agnostic (bool): Whether to treat all data as a single class.
                Defaults to `False`.
            shared_data_store (Optional[MetricDataStore]): If you have
                a hierarchy of metrics, you can pass a data store to share it
                between them, saving memory. The responsibility of updating
                the store falls on the parent metric (that contain this one).
        """
        if metric_target not in [MetricTarget.BOXES, MetricTarget.MASKS]:
            raise NotImplementedError(
                f"Intersection over union is not implemented for {metric_target}."
            )

        self._metric_target = metric_target
        self._class_agnostic = class_agnostic

        if shared_data_store:
            self._is_store_shared = True
            self._store = shared_data_store
        else:
            self._is_store_shared = False
            self._store = MetricDataStore(metric_target, class_agnostic)

    def reset(self) -> None:
        if self._is_store_shared:
            return
        self._store.reset()

    def update(
        self,
        data_1: Union[Detections, List[Detections]],
        data_2: Union[Detections, List[Detections]],
    ) -> IntersectionOverUnion:
        """
        Add data to the metric, without computing the result.
        Should call all update methods of the shared data store.

        Args:
            data_1 (Union[Detection, List[Detections]]): The first set of data.
            data_2 (Union[Detection, List[Detections]]): The second set of data.

        Returns:
            Metric: The metric object itself. You can get the metric result
            by calling the `compute` method.
        """
        if self._is_store_shared:
            # Should be updated by the parent metric
            return self

        if not isinstance(data_1, list):
            data_1 = [data_1]
        if not isinstance(data_2, list):
            data_2 = [data_2]

        for d1, d2 in zip_longest(data_1, data_2, fillvalue=Detections.empty()):
            self._update(d1, d2)

        return self

    def _update(
        self,
        data_1: Detections,
        data_2: Detections,
    ) -> None:
        assert not self._is_store_shared
        self._store.update(data_1, data_2)

    def compute(self) -> IntersectionOverUnionResult:
        """
        Compute the Intersection over Union metric (IoU)
        Uses the data set with the `update` method.

        Returns:
            Dict[int, npt.NDArray[np.float32]]: A dictionary with class IDs as keys.
            If no class ID is provided, the key is the value CLASS_ID_NONE. The values
            are (N, M) arrays where N is the number of predictions and M is the number
            of targets.
        """
        ious_by_class = {}
        for class_id in self._store.get_classes():
            if self._metric_target == MetricTarget.BOXES:
                (data_array_1, _, _), (data_array_2, _, _) = self._store.get(
                    class_id=class_id
                )
                ious_by_class[class_id] = box_iou_batch(
                    data_array_1, data_array_2
                ).transpose()
            elif self._metric_target == MetricTarget.MASKS:
                (data_array_1, _, _), (data_array_2, _, _) = self._store.get(
                    class_id=class_id
                )
                ious_by_class[class_id] = mask_iou_batch(
                    data_array_1, data_array_2
                ).transpose()
            else:
                raise NotImplementedError(
                    f"Intersection over union is not implemented"
                    f" for {self._metric_target}."
                )
        return IntersectionOverUnionResult(ious_by_class, self._metric_target)


@dataclass
class IntersectionOverUnionResult:
    ious_by_class: Dict[int, npt.NDArray[np.float32]]
    metric_target: MetricTarget

    @property
    def class_ids(self) -> List[int]:
        return list(self.ious_by_class.keys())

    def __getitem__(self, class_id: int) -> npt.NDArray[np.float32]:
        return self.ious_by_class[class_id]

    def __iter__(self):
        return iter(self.ious_by_class.items())

    def __str__(self) -> str:
        out_str = f"{self.__class__.__name__}:\n"
        for class_id, iou in self.ious_by_class.items():
            out_str += f"IoUs for class {class_id}:\n{str(iou)}\n"
        return out_str

    def to_pandas(self) -> Dict[int, "pd.DataFrame"]:
        ensure_pandas_installed()
        import pandas as pd

        return {
            class_id: pd.DataFrame(iou) for class_id, iou in self.ious_by_class.items()
        }

    def plot(self, class_id=None):
        """
        Visualize the IoU results.

        Args:
            class_id (Optional[int]): The class ID to visualize. If not
                provided, all classes will be visualized.
        """
        if class_id:
            self._plot_class(class_id)
        else:
            for cls in self.ious_by_class:
                self._plot_class(cls)

    def _plot_class(self, class_id):
        """
        Helper function to plot a single class IoU matrix or show
        zero-sized information.

        Args:
            class_id (int): The class ID to plot.
        """
        iou_matrix = self.ious_by_class[class_id]

        if iou_matrix.size == 0:
            print(
                f"No data for class {class_id}, with result shape"
                f" {iou_matrix.shape}. Skipping plot."
            )
        else:
            plt.rcParams["font.family"] = "monospace"

            colors = [
                (0, "#4D049A"),
                (0.3, "#E6194B"),
                (0.6, "#FFE119"),
                (1, "#4CFB12"),
            ]
            custom_colormap = LinearSegmentedColormap.from_list("roboflow_cmap", colors)

            row_count, col_count = iou_matrix.shape
            font_size = int(5 + (25 / (max(row_count, col_count) + 1e-6)))

            plt.figure(figsize=(6, 6))
            plt.matshow(iou_matrix, cmap=custom_colormap, fignum=1)
            plt.title(f"Class {class_id} IoU Matrix", pad=50, fontweight="bold")

            plt.gca().xaxis.set_label_position("top")
            plt.gca().xaxis.tick_top()
            plt.xlabel("Prediction index", fontweight="bold")
            plt.ylabel("Target index", fontweight="bold")
            plt.colorbar()

            for (i, j), iou_value in np.ndenumerate(iou_matrix):
                if iou_value < 0.1:
                    continue

                formatted_value = f"{iou_value:.2f}".lstrip("0")
                is_black = iou_value > 0.3
                plt.text(
                    j,
                    i,
                    formatted_value,
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    color="black" if is_black else "white",
                )

            plt.rcParams["font.family"] = "sans-serif"

            plt.show()
