"""Tests for :meth:`supervision.detection.core.Detections.remap_classes`."""

from __future__ import annotations

import numpy as np
import pytest

from supervision.detection.core import Detections


@pytest.fixture
def animals() -> Detections:
    """Four detections named dog, cat, eagle and car, with ids, scores and tracks."""
    return Detections(
        xyxy=np.array([[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3], [3, 3, 4, 4]]),
        class_id=np.array([0, 1, 2, 3]),
        confidence=np.array([0.9, 0.8, 0.7, 0.6]),
        tracker_id=np.array([10, 11, 12, 13]),
        data={"class_name": np.array(["dog", "cat", "eagle", "car"])},
    )


class TestRemapClasses:
    """Remapping and filtering of detections onto a target class list."""

    def test_remaps_column_shaped_class_names(self, animals: Detections) -> None:
        """A column of class names maps one name per detection."""
        # Arrange
        animals.data["class_name"] = np.array([["dog"], ["cat"], ["eagle"], ["car"]])

        # Act
        result = animals.remap_classes(["animal", "car"], {"dog": "animal"})

        # Assert
        assert result.class_id.tolist() == [0, 1]
        assert result.data["class_name"].tolist() == ["animal", "car"]
        assert result.tracker_id.tolist() == [10, 13]

    def test_rejects_multiple_class_names_per_detection(
        self, animals: Detections
    ) -> None:
        """Multiple names per row cannot be assigned a single class ID."""
        # Arrange
        animals.data["class_name"] = np.full((4, 2), "dog")

        # Act / Assert
        with pytest.raises(ValueError, match="class_name must have shape"):
            animals.remap_classes(["dog"])

    def test_remaps_names_and_ids(self, animals: Detections) -> None:
        """Mapped names replace the originals and ids index the target classes."""
        # Arrange
        classes = ["animal", "bird", "car"]
        mapping = {"dog": "animal", "cat": "animal", "eagle": "bird"}

        # Act
        result = animals.remap_classes(classes, class_name_mapping=mapping)

        # Assert
        assert result.data["class_name"].tolist() == ["animal", "animal", "bird", "car"]
        assert result.class_id.tolist() == [0, 0, 1, 2]

    def test_drops_detections_outside_the_classes(self, animals: Detections) -> None:
        """Without a mapping, only detections named in the target classes remain."""
        # Act
        result = animals.remap_classes(["car"])

        # Assert
        assert result.data["class_name"].tolist() == ["car"]
        assert result.class_id.tolist() == [0]

    def test_keeps_other_fields_aligned_with_kept_rows(
        self, animals: Detections
    ) -> None:
        """Boxes, scores and tracker ids follow the rows that survive the filter."""
        # Act
        result = animals.remap_classes(["car", "dog"])

        # Assert
        assert result.xyxy.tolist() == [[0, 0, 1, 1], [3, 3, 4, 4]]
        assert result.confidence.tolist() == [0.9, 0.6]
        assert result.tracker_id.tolist() == [10, 13]
        assert result.class_id.tolist() == [1, 0]

    def test_keeps_masks_of_kept_rows(self, animals: Detections) -> None:
        """Segmentation masks are filtered together with the rows."""
        # Arrange
        animals.mask = np.arange(4 * 2 * 2).reshape(4, 2, 2) % 2 == 0

        # Act
        result = animals.remap_classes(["car", "dog"])

        # Assert
        assert np.array_equal(result.mask, animals.mask[[0, 3]])

    def test_does_not_modify_the_input(self, animals: Detections) -> None:
        """The source detections keep their original ids and names."""
        # Act
        animals.remap_classes(["animal"], class_name_mapping={"dog": "animal"})

        # Assert
        assert animals.class_id.tolist() == [0, 1, 2, 3]
        assert animals.data["class_name"].tolist() == ["dog", "cat", "eagle", "car"]

    def test_keeps_list_valued_data_of_kept_rows(self, animals: Detections) -> None:
        """Per-detection lists, ragged ones included, are filtered, not coerced."""
        # Arrange
        animals.data["tag"] = ["a", "b", "c", "d"]
        animals.data["ragged"] = [[1], [1, 2], [], [3, 4, 5]]

        # Act
        result = animals.remap_classes(["car", "dog"])

        # Assert
        assert result.data["tag"] == ["a", "d"]
        assert result.data["ragged"] == [[1], [3, 4, 5]]

    def test_keeps_zero_dimensional_data(self, animals: Detections) -> None:
        """A scalar data entry is not per-detection and is carried over unchanged."""
        # Arrange
        animals.data["scale"] = np.array(5)

        # Act
        result = animals.remap_classes(["car"])

        # Assert
        assert result.data["scale"] == 5

    def test_returns_integer_ids_when_nothing_is_kept(
        self, animals: Detections
    ) -> None:
        """A fully filtered result is empty and still carries integer class ids."""
        # Act
        result = animals.remap_classes(["zebra"])

        # Assert
        assert len(result) == 0
        assert np.issubdtype(result.class_id.dtype, np.integer)

    def test_returns_integer_ids_for_empty_input(self) -> None:
        """Empty input stays empty and its class ids are integers."""
        # Arrange
        detections = Detections.empty()
        detections.data["class_name"] = np.array([], dtype=str)

        # Act
        result = detections.remap_classes(["car"])

        # Assert
        assert len(result) == 0
        assert np.issubdtype(result.class_id.dtype, np.integer)

    def test_raises_without_class_name(self) -> None:
        """Detections without a class_name entry cannot be remapped."""
        # Arrange
        detections = Detections(xyxy=np.array([[0, 0, 1, 1]]), data={})

        # Act / Assert
        with pytest.raises(ValueError, match="class_name"):
            detections.remap_classes(["car"])

    @pytest.mark.parametrize(
        ("classes", "mapping", "error", "match"),
        [
            pytest.param(
                ["car", "car"], None, ValueError, "duplicate", id="duplicates"
            ),
            pytest.param(
                ["animal"],
                {"dog": "animl"},
                ValueError,
                "missing from classes",
                id="misspelled-target",
            ),
            pytest.param(
                ["a", "b"], {0: 1}, TypeError, "class ids are not supported", id="ids"
            ),
            pytest.param(
                ["car"], ["dog"], TypeError, "must be a mapping", id="not-a-mapping"
            ),
        ],
    )
    def test_rejects_invalid_arguments(
        self,
        animals: Detections,
        classes: list[str],
        mapping: object,
        error: type[Exception],
        match: str,
    ) -> None:
        """Arguments that would silently drop or misassign detections raise."""
        # Act / Assert
        with pytest.raises(error, match=match):
            animals.remap_classes(classes, class_name_mapping=mapping)  # type: ignore[arg-type]
