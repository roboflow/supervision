from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

import supervision as sv
from supervision.annotators.base import BaseAnnotator
from supervision.utils.conversion import _opencv_can_draw_into

LAYOUTS = [
    pytest.param(3, lambda image: image[:, ::2], id="column-step"),
    pytest.param(3, lambda image: image[::-1], id="row-flip"),
    pytest.param(3, lambda image: image[:, ::-1], id="column-flip"),
    pytest.param(3, lambda image: image[..., ::-1], id="channel-flip"),
    pytest.param(3, lambda image: image.transpose(1, 0, 2), id="transpose"),
    pytest.param(4, lambda image: image[..., :3], id="bgra-color-channels"),
    pytest.param(1, lambda image: image[:, ::2], id="grayscale-column-step"),
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


def _parent(channels: int) -> npt.NDArray[np.uint8]:
    """Build a seeded random 64x64 image with 1, 3 or 4 channels."""
    shape = (64, 64) if channels == 1 else (64, 64, channels)
    return np.random.default_rng(0).integers(0, 255, size=shape, dtype=np.uint8)


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
    @pytest.mark.parametrize(("channels", "layout"), LAYOUTS)
    @pytest.mark.parametrize("make_annotator", ANNOTATORS)
    def test_view_is_annotated_in_place_like_contiguous_copy(
        self,
        make_annotator: Callable[[], BaseAnnotator],
        channels: int,
        layout: Callable[[npt.NDArray[np.uint8]], npt.NDArray[np.uint8]],
    ) -> None:
        """Annotating a non-contiguous view draws the same pixels into the view only."""
        # given
        parent = _parent(channels)
        view = layout(parent)
        detections = _detections_for(view)
        expected_parent = parent.copy()
        expected_view = layout(expected_parent)
        expected_view[...] = make_annotator().annotate(
            np.ascontiguousarray(expected_view), detections
        )

        # when
        result = make_annotator().annotate(view, detections)

        # then
        assert result is view
        np.testing.assert_array_equal(parent, expected_parent)

    def test_read_only_view_is_left_unchanged(self) -> None:
        """A read-only view is passed through and never written to."""
        # given
        parent = _parent(3)
        parent.setflags(write=False)
        view = parent[:, ::-1]
        before = parent.copy()

        # when
        with pytest.raises(Exception):  # noqa: PT011  # backend-specific error
            sv.BoxAnnotator().annotate(view, _detections_for(view))

        # then
        np.testing.assert_array_equal(parent, before)

    def test_masked_array_keeps_its_mask(self) -> None:
        """A masked-array view keeps the parent's mask when nothing is drawn."""
        # given
        parent = np.ma.masked_array(_parent(3), mask=np.zeros((64, 64, 3), bool))
        parent.mask[..., 0] = True
        mask_before = parent.mask.copy()

        # when
        sv.BoxAnnotator().annotate(parent[:, ::2], sv.Detections.empty())

        # then
        np.testing.assert_array_equal(parent.mask, mask_before)


class TestOpencvCanDrawInto:
    @pytest.mark.parametrize(
        ("image", "expected"),
        [
            pytest.param(np.zeros((8, 8, 3), np.uint8), True, id="contiguous"),
            pytest.param(np.zeros((8, 8, 3), np.uint8)[2:6, 1:5], True, id="crop"),
            pytest.param(np.zeros((8, 8, 3), np.uint8)[::2], True, id="row-step"),
            pytest.param(np.zeros((8, 8), np.uint8)[1:5, 2:6], True, id="gray-crop"),
            pytest.param(np.zeros((8, 8, 3), np.uint8)[:, ::2], False, id="col-step"),
            pytest.param(np.zeros((8, 8, 3), np.uint8)[::-1], False, id="row-flip"),
            pytest.param(np.zeros((8, 8, 3), np.uint8)[..., ::-1], False, id="bgr-rgb"),
            pytest.param(np.zeros((8, 8, 4), np.uint8)[..., :3], False, id="subset"),
            pytest.param(np.zeros((2, 8, 8, 3), np.uint8), False, id="four-dims"),
            pytest.param(
                np.ndarray(
                    (8, 8, 3),
                    np.float32,
                    buffer=np.zeros(8 * 97 + 96, np.uint8),
                    strides=(97, 12, 4),
                ),
                False,
                id="padded-float-rows",
            ),
        ],
    )
    def test_layout(self, image: npt.NDArray[np.generic], expected: bool) -> None:
        """Only layouts OpenCV can draw into directly are accepted."""
        # when
        result = _opencv_can_draw_into(image)

        # then
        assert result is expected
