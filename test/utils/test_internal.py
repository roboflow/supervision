from contextlib import ExitStack as DoesNotRaise
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.utils.internal import get_instance_variables


class MockClass:
    def __init__(self):
        self.public = 0
        self._protected = 1
        self.__private = 2

    def public_method(self):
        pass

    def _protected_method(self):
        pass

    def __private_method(self):
        pass

    @property
    def public_property(self):
        return 0

    @property
    def _protected_property(self):
        return 1

    @property
    def __private_property(self):
        return 2


@dataclass
class MockDataclass:
    public: int = 0
    _protected: int = 1
    __private: int = 2

    public_field: int = field(default=0)
    _protected_field: int = field(default=1)
    __private_field: int = field(default=2)

    public_field_with_factory: dict = field(default_factory=dict)
    _protected_field_with_factory: dict = field(default_factory=dict)
    __private_field_with_factory: dict = field(default_factory=dict)

    def public_method(self):
        pass

    def _protected_method(self):
        pass

    def __private_method(self):
        pass

    @property
    def public_property(self):
        return 0

    @property
    def _protected_property(self):
        return 1

    @property
    def __private_property(self):
        return 2


@pytest.mark.parametrize(
    "input_instance, include_properties, expected, exception",
    [
        (
            MockClass,
            False,
            None,
            pytest.raises(ValueError),
        ),
        (
            MockClass(),
            False,
            {"public"},
            DoesNotRaise(),
        ),
        (
            MockClass(),
            True,
            {"public", "public_property"},
            DoesNotRaise(),
        ),
        (
            MockDataclass(),
            False,
            {"public", "public_field", "public_field_with_factory"},
            DoesNotRaise(),
        ),
        (
            MockDataclass(),
            True,
            {"public", "public_field", "public_field_with_factory", "public_property"},
            DoesNotRaise(),
        ),
        (
            Detections,
            False,
            None,
            pytest.raises(ValueError),
        ),
        (
            Detections,
            True,
            None,
            pytest.raises(ValueError),
        ),
        (
            Detections.empty(),
            False,
            {
                "xyxy",
                "class_id",
                "confidence",
                "mask",
                "tracker_id",
                "data",
                "metadata",
            },
            DoesNotRaise(),
        ),
        (
            Detections.empty(),
            True,
            {
                "xyxy",
                "class_id",
                "confidence",
                "mask",
                "tracker_id",
                "data",
                "metadata",
                "area",
                "box_area",
            },
            DoesNotRaise(),
        ),
        (
            Detections(xyxy=np.array([[1, 2, 3, 4]])),
            False,
            {
                "xyxy",
                "class_id",
                "confidence",
                "mask",
                "tracker_id",
                "data",
                "metadata",
            },
            DoesNotRaise(),
        ),
        (
            Detections(
                xyxy=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
                class_id=np.array([1, 2]),
                confidence=np.array([0.1, 0.2]),
                mask=np.array([[[1]], [[2]]]),
                tracker_id=np.array([1, 2]),
                data={"key_1": [1, 2], "key_2": [3, 4]},
            ),
            False,
            {
                "xyxy",
                "class_id",
                "confidence",
                "mask",
                "tracker_id",
                "data",
                "metadata",
            },
            DoesNotRaise(),
        ),
        (
            Detections.empty(),
            False,
            {
                "xyxy",
                "class_id",
                "confidence",
                "mask",
                "tracker_id",
                "data",
                "metadata",
            },
            DoesNotRaise(),
        ),
    ],
)
def test_get_instance_variables(
    input_instance: Any,
    include_properties: bool,
    expected: set[str],
    exception: Exception,
) -> None:
    with exception:
        result = get_instance_variables(
            input_instance, include_properties=include_properties
        )
        assert result == expected
