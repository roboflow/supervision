from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from PIL import Image

import supervision as sv

#: Red, green and blue in RGB, so palette index 0, 1 and 2 are easy to tell apart.
PALETTE = sv.ColorPalette.from_hex(["#ff0000", "#00ff00", "#0000ff"])


def make_scene(height: int = 2, width: int = 3) -> np.ndarray:
    """Return a black BGR scene."""
    return np.zeros((height, width, 3), dtype=np.uint8)


class TestLabelMapAnnotator:
    """`LabelMapAnnotator` colours an `(H, W)` class-id map by palette index."""

    def test_paints_each_class_with_its_palette_color(self) -> None:
        """Class ids 1 and 2 take the second and third palette colours."""
        label_map = np.array([[1, 2, 1], [2, 1, 2]])
        annotator = sv.LabelMapAnnotator(color=PALETTE, opacity=1.0, background_id=None)

        result = annotator.annotate(make_scene(), label_map)

        green, blue = [0, 255, 0], [255, 0, 0]
        assert result.tolist() == [
            [green, blue, green],
            [blue, green, blue],
        ]

    def test_leaves_background_id_unpainted_by_default(self) -> None:
        """Id `0` is background by default, so those pixels keep the scene."""
        label_map = np.array([[0, 1, 0], [1, 0, 1]])
        annotator = sv.LabelMapAnnotator(color=PALETTE, opacity=1.0)

        result = annotator.annotate(make_scene(), label_map)

        assert result[..., 1].tolist() == [[0, 255, 0], [255, 0, 255]]
        assert result[..., 0].max() == 0
        assert result[..., 2].max() == 0

    def test_paints_every_id_when_background_is_none(self) -> None:
        """`background_id=None` paints id `0` with the first palette colour."""
        label_map = np.zeros((2, 3), dtype=np.int64)
        annotator = sv.LabelMapAnnotator(color=PALETTE, opacity=1.0, background_id=None)

        result = annotator.annotate(make_scene(), label_map)

        assert np.all(result == [0, 0, 255])

    def test_leaves_a_custom_background_id_unpainted(self) -> None:
        """A background id other than `0` is skipped and `0` is painted."""
        label_map = np.array([[255, 0, 255], [0, 255, 0]])
        annotator = sv.LabelMapAnnotator(color=PALETTE, opacity=1.0, background_id=255)

        result = annotator.annotate(make_scene(), label_map)

        painted = np.any(result != 0, axis=2)
        assert painted.tolist() == [[False, True, False], [True, False, True]]

    def test_leaves_negative_ids_unpainted(self) -> None:
        """A negative id is an ignore label, not a class, so it is never painted."""
        label_map = np.array([[-1, 1, -1], [1, -1, 1]])
        annotator = sv.LabelMapAnnotator(color=PALETTE, opacity=1.0, background_id=None)

        result = annotator.annotate(make_scene(), label_map)

        painted = np.any(result != 0, axis=2)
        assert painted.tolist() == [[False, True, False], [True, False, True]]

    def test_wraps_ids_beyond_the_palette(self) -> None:
        """Id `4` takes the colour of palette index `1`, as the other annotators do."""
        label_map = np.full((2, 3), 4)
        annotator = sv.LabelMapAnnotator(color=PALETTE, opacity=1.0)

        result = annotator.annotate(make_scene(), label_map)

        assert np.all(result == [0, 255, 0])

    def test_blends_the_color_over_the_scene_at_opacity(self) -> None:
        """A quarter opacity over black gives a quarter of the colour."""
        label_map = np.full((2, 3), 2)
        annotator = sv.LabelMapAnnotator(color=PALETTE, opacity=0.25)

        result = annotator.annotate(make_scene(), label_map)

        assert np.all(result == [64, 0, 0])

    @pytest.mark.parametrize("opacity", [0.0, -1.0])
    def test_draws_nothing_at_zero_or_negative_opacity(self, opacity: float) -> None:
        """An opacity of `0` or less leaves the scene as it was."""
        label_map = np.full((2, 3), 1)
        annotator = sv.LabelMapAnnotator(color=PALETTE, opacity=opacity)

        result = annotator.annotate(make_scene(), label_map)

        assert not result.any()

    def test_replaces_pixels_at_opacity_above_one(self) -> None:
        """An opacity above `1` is clamped, so it paints like `1`."""
        label_map = np.full((2, 3), 1)
        annotator = sv.LabelMapAnnotator(color=PALETTE, opacity=3.0)

        result = annotator.annotate(make_scene(), label_map)

        assert np.all(result == [0, 255, 0])

    def test_draws_in_place_and_returns_the_scene(self) -> None:
        """An array scene is modified in place, as for the other annotators."""
        scene = make_scene()
        annotator = sv.LabelMapAnnotator(color=PALETTE, opacity=1.0)

        result = annotator.annotate(scene, np.full((2, 3), 1))

        assert result is scene
        assert scene.any()

    def test_leaves_the_label_map_unchanged(self) -> None:
        """Annotating reads the label map and never writes to it."""
        label_map = np.array([[0, 1, 2], [2, 1, 0]])
        before = label_map.copy()
        annotator = sv.LabelMapAnnotator(color=PALETTE, opacity=1.0)

        annotator.annotate(make_scene(), label_map)

        assert np.array_equal(label_map, before)

    def test_returns_a_pillow_image_for_a_pillow_scene(self) -> None:
        """A Pillow scene comes back as a Pillow image with the colours drawn."""
        scene = Image.new("RGB", (3, 2))
        annotator = sv.LabelMapAnnotator(color=PALETTE, opacity=1.0)

        result = annotator.annotate(scene, np.full((2, 3), 1))

        assert isinstance(result, Image.Image)
        assert np.all(np.asarray(result) == [0, 255, 0])

    @pytest.mark.parametrize(
        "dtype", [np.uint8, np.uint16, np.int16, np.int32, np.int64, np.uint64]
    )
    def test_accepts_integer_label_maps_of_any_width(self, dtype: Any) -> None:
        """Integer maps of every common width give the same colours."""
        label_map = np.array([[0, 1, 2], [2, 1, 0]], dtype=dtype)
        annotator = sv.LabelMapAnnotator(color=PALETTE, opacity=1.0)

        result = annotator.annotate(make_scene(), label_map)

        assert result[..., 1].tolist() == [[0, 255, 0], [0, 255, 0]]

    @pytest.mark.parametrize(
        ("dtype", "first_id"),
        [
            pytest.param("int64", 2**63 - 2, id="int64-maximum"),
            pytest.param("uint64", 2**64 - 2, id="uint64-maximum"),
            pytest.param(">u8", 2**63, id="big-endian-uint64"),
        ],
    )
    def test_paints_adjacent_near_limit_ids(self, dtype: str, first_id: int) -> None:
        """Adjacent large IDs keep distinct palette indices."""
        ids = [first_id, first_id + 1]
        label_map = np.array([ids], dtype=dtype)
        annotator = sv.LabelMapAnnotator(color=PALETTE, opacity=1.0)

        result = annotator.annotate(make_scene(height=1, width=2), label_map)

        expected = [
            PALETTE.colors[class_id % len(PALETTE.colors)].as_bgr() for class_id in ids
        ]
        np.testing.assert_array_equal(result, [expected])

    def test_matches_mask_annotator_on_loaded_detections(self) -> None:
        """Painting the map equals `MaskAnnotator` on the detections it loads to."""
        label_map = np.array([[0, 1, 1, 2], [0, 1, 2, 2], [3, 3, 2, 0]])
        detections = sv.Detections.from_semantic_segmentation(label_map)
        scene = np.full((3, 4, 3), 90, dtype=np.uint8)
        mask_annotator = sv.MaskAnnotator(color=PALETTE, opacity=1.0)
        label_map_annotator = sv.LabelMapAnnotator(color=PALETTE, opacity=1.0)

        expected = mask_annotator.annotate(scene.copy(), detections)
        result = label_map_annotator.annotate(scene.copy(), label_map)

        assert np.array_equal(result, expected)


class TestLabelMapAnnotatorValidation:
    """`LabelMapAnnotator` rejects inputs it cannot colour."""

    @pytest.mark.parametrize(
        ("label_map", "message"),
        [
            pytest.param(
                np.zeros((3, 3), dtype=np.int64), "does not match", id="wrong-size"
            ),
            pytest.param(
                np.zeros((2, 3, 1), dtype=np.int64), "2-D", id="three-dimensional"
            ),
            pytest.param(np.zeros((2, 3), dtype=np.float32), "integer", id="float"),
            pytest.param(np.zeros((2, 3), dtype=bool), "integer", id="bool"),
            pytest.param(
                np.zeros((2, 3), dtype="timedelta64[D]"), "integer", id="timedelta"
            ),
            pytest.param(
                np.array([[0, 1 << 30, 0], [0, 0, 0]]), "span", id="id-span-too-wide"
            ),
        ],
    )
    def test_raises_for_an_invalid_label_map(
        self, label_map: np.ndarray, message: str
    ) -> None:
        """A map of the wrong size, rank, dtype or id span raises `ValueError`."""
        annotator = sv.LabelMapAnnotator(color=PALETTE)

        with pytest.raises(ValueError, match=message):
            annotator.annotate(make_scene(), label_map)

    @pytest.mark.parametrize(
        ("scene", "message"),
        [
            pytest.param(
                np.zeros((2, 3), dtype=np.uint8), "3-channel", id="grayscale-scene"
            ),
            pytest.param(
                np.zeros((2, 3, 4), dtype=np.uint8), "3-channel", id="four-channels"
            ),
            pytest.param(
                np.zeros((2, 3, 3), dtype=np.float32), "uint8", id="float-scene"
            ),
        ],
    )
    def test_raises_for_an_unsupported_scene(
        self, scene: np.ndarray, message: str
    ) -> None:
        """A scene that is not a 3-channel `uint8` image raises `ValueError`."""
        annotator = sv.LabelMapAnnotator(color=PALETTE)

        with pytest.raises(ValueError, match=message):
            annotator.annotate(scene, np.zeros((2, 3), dtype=np.int64))
