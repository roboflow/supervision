from __future__ import annotations

import numpy as np
import numpy.typing as npt

from supervision.detection.core import Detections
from supervision.detection.tools.inference_slicer import move_detections
from supervision.detection.utils.boxes import clip_boxes
from supervision.draw.base import ImageType
from supervision.utils.image import crop_image, get_image_resolution_wh


class TrainingSlicer:
    """
        Slice a full-resolution image and its ground-truth `Detections` into a grid
        of smaller tiles for training small-object detection or segmentation models.

        This is the training-time counterpart to `InferenceSlicer`: instead of
        running a model callback on each tile and merging predictions back into
        full-image coordinates, `TrainingSlicer` slices existing ground-truth
        annotations to match each tile, translating box and mask coordinates into
        the tile's local coordinate frame. This is the standard way to turn a
        dataset of large images into fixed-size training crops (the same technique
        used by tools like SAHI) so that small objects occupy a larger fraction of
        the input a detector sees.

        Args:
            slice_wh: Size of each tile `(width, height)`. If int, both width and
                height are set to this value.
            overlap_wh: Overlap size `(width, height)` between tiles. If int, both
                width and height are set to this value. Unlike `InferenceSlicer`,
                training tiles are typically generated without overlap; increase
                this if objects that straddle a tile boundary should also appear
                whole in a neighboring tile.
            min_visibility: Minimum fraction, in `[0, 1]`, of an annotation's
                original box area that must remain after clipping to a tile for
                that annotation to be kept in the tile. Annotations that are cut
                below this threshold by a tile boundary are dropped from that tile
                (they still appear whole in any other tile that fully contains
                them, when `overlap_wh` provides one). Defaults to `0.1`.
            drop_empty_slices: If `True`, tiles left with zero annotations after
                slicing are omitted from the returned list. If `False` (default),
                every tile is returned, including background-only tiles — useful
                for hard-negative mining. Defaults to `False`.

        Raises:
            ValueError: If `slice_wh` or `overlap_wh` are invalid or inconsistent.
            ValueError: If `min_visibility` is not in `[0, 1]`.

        Note:
            Oriented bounding boxes (`OBB`) are translated into each tile's local
            frame but are not clipped to the tile rectangle, since clipping a
            rotated box against an axis-aligned rectangle produces a polygon with
            more than four vertices. A box that straddles a tile boundary may
            therefore have vertices outside that tile. Axis-aligned boxes and
            masks are always clipped/cropped to the tile.

        Example:
    ```python
            import supervision as sv
            from supervision import _cv2 as cv2

            image = cv2.imread("large_train_image.jpg")
            detections = sv.Detections(...)  # ground truth for the full image

            slicer = sv.TrainingSlicer(slice_wh=320, overlap_wh=0)
            for tile_image, tile_detections in slicer(image, detections):
                ...  # write out one training sample per tile
    ```
    """

    def __init__(
        self,
        slice_wh: int | tuple[int, int] = 320,
        overlap_wh: int | tuple[int, int] = 0,
        min_visibility: float = 0.1,
        drop_empty_slices: bool = False,
    ):
        slice_wh_norm = self._normalize_wh_pair(slice_wh, "slice_wh", allow_zero=False)
        overlap_wh_norm = self._normalize_wh_pair(
            overlap_wh, "overlap_wh", allow_zero=True
        )
        self._validate_overlap(slice_wh=slice_wh_norm, overlap_wh=overlap_wh_norm)

        if not 0.0 <= min_visibility <= 1.0:
            raise ValueError(
                "`min_visibility` must be in the range [0, 1]. "
                f"Received: {min_visibility}"
            )

        self.slice_wh = slice_wh_norm
        self.overlap_wh = overlap_wh_norm
        self.min_visibility = min_visibility
        self.drop_empty_slices = drop_empty_slices

    def __call__(
        self, image: ImageType, detections: Detections
    ) -> list[tuple[ImageType, Detections]]:
        """
        Slice `image` and `detections` into a grid of tiles.

        Args:
            image: The full-resolution image to slice.
            detections: Ground-truth detections for `image`, in full-image
                coordinates.

        Returns:
            A list of `(tile_image, tile_detections)` pairs, one per tile, in
            row-major order. `tile_detections` coordinates are local to
            `tile_image` (top-left of the tile is `(0, 0)`).
        """
        resolution_wh = get_image_resolution_wh(image)
        offsets = self._generate_offsets(
            resolution_wh=resolution_wh,
            slice_wh=self.slice_wh,
            overlap_wh=self.overlap_wh,
        )

        results: list[tuple[ImageType, Detections]] = []
        for offset in offsets:
            tile_image = crop_image(image=image, xyxy=offset)
            tile_detections = self._slice_detections(detections, offset)
            if self.drop_empty_slices and len(tile_detections) == 0:
                continue
            results.append((tile_image, tile_detections))
        return results

    def _slice_detections(
        self, detections: Detections, offset: npt.NDArray[np.number]
    ) -> Detections:
        """Filter `detections` to those visible in `offset` and localize them.

        Args:
            offset: Tile coordinates `(x_min, y_min, x_max, y_max)` in the
                full image's coordinate system.

        Returns:
            Detections whose coordinates are local to the tile, i.e. relative
            to `(offset[0], offset[1])`.
        """
        x_min, y_min, x_max, y_max = offset
        tile_w = int(x_max - x_min)
        tile_h = int(y_max - y_min)

        if len(detections) == 0:
            keep_mask = np.zeros(0, dtype=bool)
        else:
            xyxy = detections.xyxy
            inter_x1 = np.maximum(xyxy[:, 0], x_min)
            inter_y1 = np.maximum(xyxy[:, 1], y_min)
            inter_x2 = np.minimum(xyxy[:, 2], x_max)
            inter_y2 = np.minimum(xyxy[:, 3], y_max)
            inter_area = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(
                inter_y2 - inter_y1, 0, None
            )

            box_w = np.clip(xyxy[:, 2] - xyxy[:, 0], 0, None)
            box_h = np.clip(xyxy[:, 3] - xyxy[:, 1], 0, None)
            original_area = box_w * box_h

            with np.errstate(divide="ignore", invalid="ignore"):
                visibility = np.where(
                    original_area > 0, inter_area / original_area, 0.0
                )

            keep_mask = (inter_area > 0) & (visibility >= self.min_visibility)

        localized = move_detections(
            detections=detections.select(keep_mask),
            offset=np.array([-x_min, -y_min]),
            resolution_wh=(tile_w, tile_h),
        )
        localized.xyxy = clip_boxes(localized.xyxy, resolution_wh=(tile_w, tile_h))
        return localized

    @staticmethod
    def _normalize_wh_pair(
        value: int | tuple[int, int], name: str, allow_zero: bool
    ) -> tuple[int, int]:
        lower_bound = 0 if allow_zero else 1
        comparator = "non negative" if allow_zero else "positive"

        if isinstance(value, int):
            if value < lower_bound:
                raise ValueError(
                    f"`{name}` must be a {comparator} integer. Received: {value}"
                )
            return value, value

        if isinstance(value, tuple) and len(value) == 2:
            width, height = value
            if width < lower_bound or height < lower_bound:
                raise ValueError(
                    f"`{name}` values must be {comparator}. Received: {value}"
                )
            return width, height

        raise ValueError(
            f"`{name}` must be an int or a tuple of two {comparator} integers. "
            f"Received: {value}"
        )

    @staticmethod
    def _validate_overlap(
        slice_wh: tuple[int, int], overlap_wh: tuple[int, int]
    ) -> None:
        overlap_w, overlap_h = overlap_wh
        slice_w, slice_h = slice_wh
        if overlap_w >= slice_w or overlap_h >= slice_h:
            raise ValueError(
                "`overlap_wh` must be smaller than `slice_wh` in both dimensions "
                f"to keep a positive stride. Received overlap_wh={overlap_wh}, "
                f"slice_wh={slice_wh}."
            )

    @staticmethod
    def _generate_offsets(
        resolution_wh: tuple[int, int],
        slice_wh: tuple[int, int],
        overlap_wh: tuple[int, int],
    ) -> npt.NDArray[np.number]:
        """
        Generate the coordinates of image tiles with overlap.

        This mirrors `InferenceSlicer._generate_offset`. The two classes slice
        images for different purposes (live model inference vs. offline
        dataset preparation) and are kept independent so each can evolve
        without risking the other's behavior; the grid math itself is small,
        stable, and unit-tested in both places.

        Args:
            resolution_wh: Image resolution `(width, height)`.
            slice_wh: Size of each tile `(width, height)`.
            overlap_wh: Overlap size between tiles `(width, height)`.

        Returns:
            Array of shape `(num_slices, 4)` with each row as
                `(x_min, y_min, x_max, y_max)` coordinates for a tile.
        """
        slice_width, slice_height = slice_wh
        image_width, image_height = resolution_wh
        overlap_width, overlap_height = overlap_wh

        stride_x = slice_width - overlap_width
        stride_y = slice_height - overlap_height

        def _compute_axis_starts(
            image_size: int, slice_size: int, stride: int
        ) -> list[int]:
            if image_size <= slice_size:
                return [0]

            if stride == slice_size:
                return list(np.arange(0, image_size, stride).tolist())

            last_start = image_size - slice_size
            starts: list[int] = list(np.arange(0, last_start, stride).tolist())
            if not starts or starts[-1] != last_start:
                starts.append(last_start)
            return starts

        x_starts = _compute_axis_starts(
            image_size=image_width, slice_size=slice_width, stride=stride_x
        )
        y_starts = _compute_axis_starts(
            image_size=image_height, slice_size=slice_height, stride=stride_y
        )

        x_min, y_min = np.meshgrid(x_starts, y_starts)
        x_max = np.clip(x_min + slice_width, 0, image_width)
        y_max = np.clip(y_min + slice_height, 0, image_height)

        offsets: npt.NDArray[np.number] = np.stack(
            [x_min, y_min, x_max, y_max], axis=-1
        ).reshape(-1, 4)

        return offsets
