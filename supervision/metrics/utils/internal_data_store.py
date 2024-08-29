from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.metrics.core import CLASS_ID_NONE, MetricTarget
from supervision.metrics.utils.object_size import (
    ObjectSizeCategory,
    get_object_size_category,
)
from supervision.metrics.utils.utils import unify_pad_masks_shape

if TYPE_CHECKING:
    from supervision.detection.core import Detections


class MetricData:
    """
    A container for detection contents, decouple from `Detections`.
    """

    def __init__(self, metric_target: MetricTarget, class_agnostic: bool = False):
        self._metric_target = metric_target
        self._class_agnostic = class_agnostic
        self._content_list: List[npt.NDArray] = []
        self._confidence_list: List[npt.NDArray[np.float32]] = []
        self._class_id_list: List[npt.NDArray[np.int_]] = []

    def update(self, detections: Detections):
        """Add new detections to the store."""
        if detections.is_empty() or len(detections) == 0:
            return

        # Relies on Detections to ensure that member vars are equal in length
        # or None/empty.
        self._validate_new_entry(detections)

        new_content = self._detections_content(detections)
        self._validate_shape(new_content)
        self._content_list.append(new_content)

        if detections.confidence is not None:
            self._confidence_list.append(detections.confidence)

        if detections.class_id is not None:
            if self._class_agnostic:
                self._class_id_list.append(
                    np.full(len(detections.class_id), CLASS_ID_NONE, dtype=int)
                )
            else:
                self._class_id_list.append(detections.class_id)

    def get_classes(self) -> Set[int]:
        """Return all class IDs."""
        if self._class_agnostic:
            return {CLASS_ID_NONE}
        return set.union(*[set(class_id) for class_id in self._class_id_list])

    def get(
        self, class_id: Optional[int] = None, size_category=ObjectSizeCategory.ANY
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Get the contents, class_ids and confidences, optionally filtered by
        class_id and/or size.

        Args:
            class_id (Optional[int]): The class ID for the data to retrieve
            size_category (ObjectSizeCategory): The size of the objects to retrieve.

        Returns:
            (np.ndarray): Boxes, masks or obb, all stacked in a single array.
        """
        if self._metric_target in [
            MetricTarget.BOXES,
            MetricTarget.ORIENTED_BOUNDING_BOXES,
        ]:
            self._merge_boxes()
            if len(self._content_list) == 0:
                content = self._make_empty_content()
            else:
                content = self._content_list[0]

        elif self._metric_target == MetricTarget.MASKS:
            self._merge_masks()
            if len(self._content_list) == 0:
                content = self._make_empty_content()
            else:
                content = self._content_list[0]

        size_mask = np.full(len(content), True)
        if size_category != ObjectSizeCategory.ANY:
            sizes = get_object_size_category(content, self._metric_target)
            size_mask = sizes == size_category.value

        class_mask = np.full(len(content), True)
        if class_id is not None:
            self._merge_class_id()
            class_ids = self._class_id_list[0]
            class_mask = class_ids == class_id

        content = content[(size_mask & class_mask)]

        self._merge_class_id()
        if len(self._class_id_list) == 0:
            class_ids = np.array([], dtype=int)
        else:
            class_ids = self._class_id_list[0]
        if len(class_ids) > 0:
            class_ids = class_ids[(size_mask & class_mask)]

        self._merge_confidence()
        if len(self._confidence_list) == 0:
            confidences = np.array([], dtype=float)
        else:
            confidences = self._confidence_list[0]
        if len(confidences) > 0:
            confidences = confidences[(size_mask & class_mask)]

        return content, class_ids, confidences

    def get_class_id(self) -> npt.NDArray[np.int_]:
        self._merge_class_id()
        if len(self._class_id_list) == 0:
            return np.array([], dtype=int)
        return self._class_id_list[0]

    def get_confidence(self) -> npt.NDArray[np.float32]:
        self._merge_confidence()
        if len(self._confidence_list) == 0:
            return np.array([], dtype=float)
        return self._confidence_list[0]

    def _detections_content(self, detections: Detections) -> npt.NDArray:
        """Return boxes, masks or oriented bounding boxes from detections."""
        if self._metric_target == MetricTarget.BOXES:
            return detections.xyxy
        if self._metric_target == MetricTarget.MASKS:
            return (
                detections.mask
                if detections.mask is not None
                else self._make_empty_content()
            )
        if self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
            if obb := detections.data.get(ORIENTED_BOX_COORDINATES):
                return np.ndarray(obb, dtype=np.float32)
            return self._make_empty_content()
        raise ValueError(f"Invalid metric target: {self._metric_target}")

    def _make_empty_content(self) -> npt.NDArray:
        if self._metric_target == MetricTarget.BOXES:
            return np.empty((0, 4), dtype=np.float32)
        if self._metric_target == MetricTarget.MASKS:
            return np.empty((0, 0, 0), dtype=bool)
        if self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
            return np.empty((0, 8), dtype=np.float32)
        raise ValueError(f"Invalid metric target: {self._metric_target}")

    def _validate_new_entry(self, detections: Detections) -> None:
        if (
            len(self._content_list) == 0
            or detections.is_empty()
            or len(detections) == 0
        ):
            return

        is_self_class_empty = len(self._class_id_list) == 0
        is_self_confidence_empty = len(self._confidence_list) == 0
        is_detection_class_empty = (
            detections.class_id is None or len(detections.class_id) == 0
        )
        is_detection_confidence_empty = (
            detections.confidence is None or len(detections.confidence) == 0
        )

        if is_self_confidence_empty and not is_detection_confidence_empty:
            raise ValueError(
                "Previously stored detections without confidence,"
                " but new detections have it."
            )
        if is_self_class_empty and not is_detection_class_empty:
            raise ValueError(
                "Previously stored detections without class ID,"
                " but new detections have it."
            )
        if not is_self_class_empty and is_detection_class_empty:
            raise ValueError(
                "Started storing detections with class ID, but"
                " new detections do not have it."
            )
        if not is_self_confidence_empty and is_detection_confidence_empty:
            raise ValueError(
                "Started storing detections with confidence,"
                " but new detections do not have it."
            )

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

    def _merge_boxes(self):
        """Merge all boxes into a single array."""
        if self._metric_target not in (
            MetricTarget.BOXES,
            MetricTarget.ORIENTED_BOUNDING_BOXES,
        ):
            raise ValueError("Invalid metric target for merging boxes")
        if len(self._content_list) < 2:
            return
        self._content_list = [np.vstack(self._content_list)]

    def _merge_masks(self):
        """Merge all masks into a single array."""
        if self._metric_target != MetricTarget.MASKS:
            raise ValueError("Invalid metric target for merging masks")
        if len(self._content_list) < 2:
            return

        mew_mask_list = unify_pad_masks_shape(*self._content_list)
        self._content_list = [np.vstack(mew_mask_list)]

    def _merge_class_id(self):
        """Merge all class IDs into a single array."""
        if len(self._class_id_list) < 2:
            return
        self._class_id_list = [np.hstack(self._class_id_list)]

    def _merge_confidence(self):
        """Merge all confidences into a single array."""
        if len(self._confidence_list) < 2:
            return
        self._confidence_list = [np.hstack(self._confidence_list)]


class MetricDataStore:
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

    def get(
        self,
        class_id: Optional[int] = None,
        size_category: ObjectSizeCategory = ObjectSizeCategory.ANY,
    ) -> Tuple[
        Tuple[npt.NDArray, npt.NDArray, npt.NDArray],
        Tuple[npt.NDArray, npt.NDArray, npt.NDArray],
    ]:
        """
        Get the data for both sets, optionally filtered by class_id and/or size.
        """

        data_1, class_id_1, confidence_1 = self._data_1.get(class_id, size_category)
        data_2, class_id_2, confidence_2 = self._data_2.get(class_id, size_category)

        if self._metric_target == MetricTarget.MASKS:
            data_1, data_2 = unify_pad_masks_shape(data_1, data_2)

        return (
            (data_1, class_id_1, confidence_1),
            (data_2, class_id_2, confidence_2),
        )

    def get_classes(self) -> Set[int]:
        """Return all class IDs."""
        return self._data_1.get_classes() | self._data_2.get_classes()
