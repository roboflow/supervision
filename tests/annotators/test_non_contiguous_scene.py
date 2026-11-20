from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

import supervision as sv
from supervision.annotators.base import BaseAnnotator

LAYOUTS = [
    pytest.param(lambda image: image[:, ::2], id="column-step"),
    pytest.param(lambda image: image[::-1], id="row-flip"),
    pytest.param(lambda image: image[:, ::-1], id="column-flip"),
    pytest.param(lambda image: image[..., ::-1], id="channel-flip"),
    pytest.param(lambda image: image.transpose(1, 0, 2), id="transpose"),
]

ANNOTATORS = [
    pytest.param(sv.BoxAnnotator, id="box"),
    pytest.param(sv.RoundBoxAnnotator, id="round-box"),
    pytest.param(sv.BoxCornerAnnotator, id="box-corner"),
    pytest.param(sv.ColorAnnotator, id="color"),
    pytest.param(sv.CircleAnnotator, id="circle"),
    pytest.param(sv.DotAnnotator, id="dot"),
    pytest.param(sv.TriangleAnnotator, id="triangle"),
    pytest.param(sv.EllipseAnnotator, id="ellipse"),
    pytest.param(sv.MaskAnnotator, id="mask"),
    pytest.param(sv.PolygonAnnotator, id="polygon"),
    pytest.param(sv.HaloAnnotator, id="halo"),
    pytest.param(sv.LabelAnnotator, id="label"),
    pytest.param(sv.BlurAnnotator, id="blur"),
    pytest.param(sv.PixelateAnnotator, id="pixelate"),
    pytest.param(sv.PercentageBarAnnotator, id="percentage-bar"),
    pytest.param(sv.TraceAnnotator, id="trace"),
    pytest.param(sv.HeatMapAnnotator, id="heat-map"),
]


def _detections_for(scene: npt.NDArray[np.uint8]) -> sv.Detections:
    """Build one detection with a mask that matches the scene's height and width."""
    height, width = scene.shape[:2]
    mask = np.zeros((1, height, width), dtype=bool)
    mask[0, 8:24, 6:20] = True
    return sv.Detections(
        xyxy=np.array([[6.0, 8.0, 20.0, 24.0]]),
        mask=mask,
        class_id=np.array([0]),
        confidence=np.array([0.5]),
        tracker_id=np.array([1]),
    )


class TestAnnotateNonContiguousScene:
    @pytest.mark.parametrize("layout", LAYOUTS)
    @pytest.mark.parametrize("make_annotator", ANNOTATORS)
    def test_view_is_annotated_in_place_like_contiguous_copy(
        self,
        make_annotator: Callable[[], BaseAnnotator],
        layout: Callable[[npt.NDArray[np.uint8]], npt.NDArray[np.uint8]],
    ) -> None:
        """Annotating a non-contiguous view draws the same pixels into the view."""
        # given
        parent = np.random.default_rng(0).integers(
            0, 255, size=(64, 64, 3), dtype=np.uint8
        )
        view = layout(parent)
        detections = _detections_for(view)
        expected = make_annotator().annotate(np.ascontiguousarray(view), detections)

        # when
        result = make_annotator().annotate(view, detections)

        # then
        assert result is view
        np.testing.assert_array_equal(view, expected)
        np.testing.assert_array_equal(layout(parent), expected)
