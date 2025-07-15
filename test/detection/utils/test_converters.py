from __future__ import annotations

import numpy as np
import pytest

from supervision.detection.utils.converters import xywh_to_xyxy, xyxy_to_xywh, \
    xyxy_to_xcycarh, xcycwh_to_xyxy


@pytest.mark.parametrize(
    "xywh, expected_result",
    [
        (np.array([[10, 20, 30, 40]]), np.array([[10, 20, 40, 60]])),  # standard case
        (np.array([[0, 0, 0, 0]]), np.array([[0, 0, 0, 0]])),  # zero size bounding box
        (
            np.array([[50, 50, 100, 100]]),
            np.array([[50, 50, 150, 150]]),
        ),  # large bounding box
        (
            np.array([[-10, -20, 30, 40]]),
            np.array([[-10, -20, 20, 20]]),
        ),  # negative coordinates
        (np.array([[50, 50, 0, 30]]), np.array([[50, 50, 50, 80]])),  # zero width
        (np.array([[50, 50, 20, 0]]), np.array([[50, 50, 70, 50]])),  # zero height
        (np.array([]).reshape(0, 4), np.array([]).reshape(0, 4)),  # empty array
    ],
)
def test_xywh_to_xyxy(xywh: np.ndarray, expected_result: np.ndarray) -> None:
    result = xywh_to_xyxy(xywh)
    np.testing.assert_array_equal(result, expected_result)


@pytest.mark.parametrize(
    "xyxy, expected_result",
    [
        (np.array([[10, 20, 40, 60]]), np.array([[10, 20, 30, 40]])),  # standard case
        (np.array([[0, 0, 0, 0]]), np.array([[0, 0, 0, 0]])),  # zero size bounding box
        (
            np.array([[50, 50, 150, 150]]),
            np.array([[50, 50, 100, 100]]),
        ),  # large bounding box
        (
            np.array([[-10, -20, 20, 20]]),
            np.array([[-10, -20, 30, 40]]),
        ),  # negative coordinates
        (np.array([[50, 50, 50, 80]]), np.array([[50, 50, 0, 30]])),  # zero width
        (np.array([[50, 50, 70, 50]]), np.array([[50, 50, 20, 0]])),  # zero height
        (np.array([]).reshape(0, 4), np.array([]).reshape(0, 4)),  # empty array
    ],
)
def test_xyxy_to_xywh(xyxy: np.ndarray, expected_result: np.ndarray) -> None:
    result = xyxy_to_xywh(xyxy)
    np.testing.assert_array_equal(result, expected_result)


@pytest.mark.parametrize(
    "xyxy, expected_result",
    [
        # Empty and zero cases
        (np.array([]).reshape(0, 4), np.array([]).reshape(0, 4)),  # empty array
        (
            np.array([[0, 0, 0, 0]]),
            np.array([[0, 0, 0.0, 0]]),
        ),  # zero size bounding box
        (
            np.array([[10, 10, 10, 10]]),
            np.array([[10, 10, 0.0, 0]]),
        ),  # point (x1=x2, y1=y2)
        # Zero width/height cases
        (np.array([[50, 50, 80, 50]]), np.array([[65, 50, 0.0, 0]])),  # zero height
        (np.array([[50, 50, 50, 80]]), np.array([[50, 65, 0.0, 30]])),  # zero width
        # Standard cases
        (np.array([[10, 20, 40, 60]]), np.array([[25, 40, 0.75, 40]])),  # standard case
        (
            np.array([[-30, -40, -10, -20]]),
            np.array([[-20, -30, 1.0, 20]]),
        ),  # all negative values
        (
            np.array([[0.1, 0.2, 0.4, 0.6]]),
            np.array([[0.25, 0.4, 0.75, 0.4]]),
        ),  # values between 0-1
        # Different aspect ratios
        (
            np.array([[10, 20, 50, 100]]),
            np.array([[30, 60, 0.5, 80]]),
        ),  # tall rectangle (height > width)
        (
            np.array([[20, 10, 100, 50]]),
            np.array([[60, 30, 2.0, 40]]),
        ),  # wide rectangle (width > height)
        (
            np.array([[50, 50, 150, 150]]),
            np.array([[100, 100, 1.0, 100]]),
        ),  # height == width
        # Multiple boxes in one array
        (
            np.array([[0, 0, 0, 0], [10, 20, 40, 60]]),
            np.array([[0, 0, 0.0, 0], [25, 40, 0.75, 40]]),
        ),  # one zero-sized box and one normal box
    ],
)
def test_xyxy_to_xcycarh(xyxy: np.ndarray, expected_result: np.ndarray) -> None:
    result = xyxy_to_xcycarh(xyxy)
    np.testing.assert_allclose(result, expected_result)


@pytest.mark.parametrize(
    "xcycwh, expected_result",
    [
        (np.array([[50, 50, 20, 30]]), np.array([[40, 35, 60, 65]])),  # standard case
        (np.array([[0, 0, 0, 0]]), np.array([[0, 0, 0, 0]])),  # zero size bounding box
        (
            np.array([[50, 50, 100, 100]]),
            np.array([[0, 0, 100, 100]]),
        ),  # large bounding box centered at (50, 50)
        (
            np.array([[-10, -10, 20, 30]]),
            np.array([[-20, -25, 0, 5]]),
        ),  # negative coordinates
        (np.array([[50, 50, 0, 30]]), np.array([[50, 35, 50, 65]])),  # zero width
        (np.array([[50, 50, 20, 0]]), np.array([[40, 50, 60, 50]])),  # zero height
        (np.array([]).reshape(0, 4), np.array([]).reshape(0, 4)),  # empty array
    ],
)
def test_xcycwh_to_xyxy(xcycwh: np.ndarray, expected_result: np.ndarray) -> None:
    result = xcycwh_to_xyxy(xcycwh)
    np.testing.assert_array_equal(result, expected_result)