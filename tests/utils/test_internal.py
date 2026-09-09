import json
import os
import subprocess
import sys
import warnings
from contextlib import ExitStack as DoesNotRaise
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.utils.internal import (
    SupervisionWarnings,
    format_warning,
    get_instance_variables,
)


class MockClass:
    def __init__(self) -> None:
        self.public = 0
        self._protected = 1
        self.__private = 2

    def public_method(self) -> None:
        pass

    def _protected_method(self) -> None:
        pass

    def __private_method(self) -> None:
        pass

    @property
    def public_property(self) -> int:
        return 0

    @property
    def _protected_property(self) -> int:
        return 1

    @property
    def __private_property(self) -> int:
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

    def public_method(self) -> None:
        pass

    def _protected_method(self) -> None:
        pass

    def __private_method(self) -> None:
        pass

    @property
    def public_property(self) -> int:
        return 0

    @property
    def _protected_property(self) -> int:
        return 1

    @property
    def __private_property(self) -> int:
        return 2


def _warning_messages_for_env(
    *, new_env: str | None, legacy_env: str | None
) -> list[str]:
    """Run the deprecation-warning path under a controlled environment."""
    env = os.environ.copy()
    env.pop("SUPERVISION_DEPRECATION_WARNING", None)
    env.pop("SUPERVISON_DEPRECATION_WARNING", None)
    if new_env is not None:
        env["SUPERVISION_DEPRECATION_WARNING"] = new_env
    if legacy_env is not None:
        env["SUPERVISON_DEPRECATION_WARNING"] = legacy_env

    script = """
import json
import warnings
from supervision.utils.internal import warn_deprecated

with warnings.catch_warnings(record=True) as recorded:
    warn_deprecated("deprecated")

print(json.dumps([str(item.message) for item in recorded]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(completed.stdout)


@pytest.mark.parametrize(
    ("input_instance", "include_properties", "expected", "exception"),
    [
        (
            MockClass,
            False,
            None,
            pytest.raises(ValueError, match="Only class instances are supported"),
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
            pytest.raises(ValueError, match="Only class instances are supported"),
        ),
        (
            Detections,
            True,
            None,
            pytest.raises(ValueError, match="Only class instances are supported"),
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
                "box_aspect_ratio",
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
                mask=np.array([[[1]], [[2]]], dtype=bool),
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


def test_supervision_warning_formatter_is_not_global() -> None:
    """Supervision warning formatting is opt-in, not global warnings state."""
    assert warnings.formatwarning is not format_warning
    assert format_warning("message", SupervisionWarnings, "file.py", 1) == (
        "SupervisionWarnings: message\n"
    )


@pytest.mark.parametrize(
    ("new_env", "legacy_env", "expected_count"),
    [
        pytest.param("0", "1", 0, id="prefer-new-env"),
        pytest.param(None, "0", 0, id="legacy-fallback"),
        pytest.param("1", "0", 1, id="new-env-enables-warning"),
    ],
)
def test_supervision_warning_env_var_controls_deprecation_warnings(
    new_env: str | None, legacy_env: str | None, expected_count: int
) -> None:
    """SUPERVISION_DEPRECATION_WARNING keeps precedence over the legacy alias."""
    messages = _warning_messages_for_env(new_env=new_env, legacy_env=legacy_env)
    assert len(messages) == expected_count
