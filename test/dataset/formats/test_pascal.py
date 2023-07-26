from contextlib import ExitStack as DoesNotRaise
from typing import List, Optional, Tuple

import numpy as np
import pytest

from supervision.dataset.formats.pascal_voc import (
    detections_to_pascal_voc,
    load_pascal_voc_annotations,
    object_to_pascal_voc,
)
from supervision.detection.core import Detections

# TODO


def test_detections_to_pascal_voc(
    expected_result, exception: Exception
):
    ...


def test_load_pascal_voc_annotations(
    expected_result, exception: Exception
):
    ...


def test_object_to_pascal_voc(
    expected_result, exception: Exception
):
    ...
