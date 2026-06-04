from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal, cast

import cv2
import numpy as np
import numpy.typing as npt

from supervision.detection.utils.boxes import pad_boxes, spread_out_boxes
from supervision.draw.base import ImageType
from supervision.draw.color import Color
from supervision.draw.utils import draw_rounded_rectangle
from supervision.geometry.core import Rect
from supervision.key_points.core import KeyPoints
from supervision.key_points.skeletons import SKELETONS_BY_VERTEX_COUNT
from supervision.utils.conversion import ensure_cv2_image_for_class_method
from supervision.utils.logger import _get_logger

logger = _get_logger(__name__)


class BaseKeyPointAnnotator(ABC):
    @abstractmethod
    def annotate(self, scene: ImageType, key_points: KeyPoints) -> ImageType:
        pass


class VertexAnnotator(BaseKeyPointAnnotator):
    """
    A class that specializes in drawing skeleton vertices on images. It uses
    specified key points to determine the locations where the vertices should be
    drawn.
    """

    def __init__(
        self,
        color: Color = Color.ROBOFLOW,
        radius: int = 4,
    ) -> None:
        """
        Args:
            color: The color to use for annotating key points.
            radius: The radius of the circles used to represent the key points.
        """
        self.color = color
        self.radius = radius

    @ensure_cv2_image_for_class_method
    def annotate(self, scene: ImageType, key_points: KeyPoints) -> ImageType:
        """
        Annotates the given scene with skeleton vertices based on the provided key
        points. It draws circles at each key point location.

        Args:
            scene: The image where skeleton vertices will be drawn. `ImageType` is a
                flexible type, accepting either `numpy.ndarray` or `PIL.Image.Image`.
            key_points: A collection of key points where each key point consists of x
                and y coordinates.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
            >>> key_points = sv.KeyPoints(
            ...     xy=np.array([[[50, 50], [60, 60]]], dtype=np.float32)
            ... )
            >>> vertex_annotator = sv.VertexAnnotator(
            ...     color=sv.Color.GREEN,
            ...     radius=10
            ... )
            >>> annotated_frame = vertex_annotator.annotate(
            ...     scene=image.copy(),
            ...     key_points=key_points
            ... )
            >>> annotated_frame.shape
            (100, 100, 3)

            ```
        """
        assert isinstance(scene, np.ndarray)
        if len(key_points) == 0:
            return scene

        for xy in key_points.xy:
            for x, y in xy:
                cv2.circle(
                    img=scene,
                    center=(int(x), int(y)),
                    radius=self.radius,
                    color=self.color.as_bgr(),
                    thickness=-1,
                )

        return scene


class EdgeAnnotator(BaseKeyPointAnnotator):
    """
    A class that specializes in drawing skeleton edges on images using specified key
    points. It connects key points with lines to form the skeleton structure.
    """

    def __init__(
        self,
        color: Color = Color.ROBOFLOW,
        thickness: int = 2,
        edges: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        """
        Args:
            color: The color to use for the edges.
            thickness: The thickness of the edges.
            edges: The edges to draw. If set to `None`, will attempt to select
                automatically.
        """
        self.color = color
        self.thickness = thickness
        self.edges = edges

    @ensure_cv2_image_for_class_method
    def annotate(self, scene: ImageType, key_points: KeyPoints) -> ImageType:
        """
        Annotates the given scene by drawing lines between specified key points to form
        edges.

        Args:
            scene: The image where skeleton edges will be drawn. `ImageType` is a
                flexible type, accepting either `numpy.ndarray` or `PIL.Image.Image`.
            key_points: A collection of key points where each key point consists of x
                and y coordinates.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
            >>> key_points = sv.KeyPoints(
            ...     xy=np.array([[[50, 50], [60, 60]]], dtype=np.float32)
            ... )
            >>> edge_annotator = sv.EdgeAnnotator(
            ...     color=sv.Color.GREEN,
            ...     thickness=5
            ... )
            >>> annotated_frame = edge_annotator.annotate(
            ...     scene=image.copy(),
            ...     key_points=key_points
            ... )
            >>> annotated_frame.shape
            (100, 100, 3)

            ```
        """
        assert isinstance(scene, np.ndarray)
        if len(key_points) == 0:
            return scene

        for xy in key_points.xy:
            edges = self.edges
            if not edges:
                edges = SKELETONS_BY_VERTEX_COUNT.get(len(xy))
            if not edges:
                logger.warning("No skeleton found with %d vertices", len(xy))
                return scene

            for class_a, class_b in edges:
                xy_a = xy[class_a - 1]
                xy_b = xy[class_b - 1]
                missing_a = np.allclose(xy_a, 0)
                missing_b = np.allclose(xy_b, 0)
                if missing_a or missing_b:
                    continue

                cv2.line(
                    img=scene,
                    pt1=(int(xy_a[0]), int(xy_a[1])),
                    pt2=(int(xy_b[0]), int(xy_b[1])),
                    color=self.color.as_bgr(),
                    thickness=self.thickness,
                )

        return scene


class VertexEllipseAnnotator(BaseKeyPointAnnotator):
    """
    A class that draws covariance ellipses around skeleton vertices.

    The annotator expects per-keypoint covariance matrices stored in
    ``key_points.data[covariance_data_key]`` with shape ``(N, K, 2, 2)`` in pixel
    coordinates, where ``N`` is the number of keypoint sets and ``K`` is the
    number of vertices per set.
    """

    def __init__(
        self,
        color: Color = Color.ROBOFLOW,
        thickness: int = 2,
        sigma: float = 2.0,
        covariance_data_key: str = "covariance",
        max_axis_length: float | None = None,
        line_style: Literal["solid", "dashed"] = "solid",
        dash_length: int = 16,
    ) -> None:
        """
        Args:
            color: The color to use for covariance ellipses.
            thickness: The line thickness used to draw the ellipses.
            sigma: Number of standard deviations represented by the ellipse axes.
            covariance_data_key: Key in ``key_points.data`` containing covariance
                matrices with shape ``(N, K, 2, 2)``.
            max_axis_length: Optional cap for ellipse semi-axis lengths in pixels.
                When ``None`` (default), near-singular precision matrices can produce
                extremely large eigenvalues and frame-spanning ellipses. Set this to
                ``min(image_height, image_width)`` or a similar bound for production
                use.
            line_style: Ellipse line style. Use ``"dashed"`` for less visually
                dominant uncertainty overlays.
            dash_length: Arc length in degrees for each dashed segment. Only used
                when ``line_style="dashed"``.
        """
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if thickness <= 0:
            raise ValueError("thickness must be positive")
        if max_axis_length is not None and max_axis_length <= 0:
            raise ValueError("max_axis_length must be positive when provided")
        if line_style not in {"solid", "dashed"}:
            raise ValueError("line_style must be 'solid' or 'dashed'")
        if dash_length <= 0:
            raise ValueError("dash_length must be positive")

        self.color = color
        self.thickness = thickness
        self.sigma = sigma
        self.covariance_data_key = covariance_data_key
        self.max_axis_length = max_axis_length
        self.line_style = line_style
        self.dash_length = dash_length

    @ensure_cv2_image_for_class_method
    def annotate(self, scene: ImageType, key_points: KeyPoints) -> ImageType:
        """
        Annotates the given scene with covariance ellipses around keypoints.

        Args:
            scene: The image where covariance ellipses will be drawn. ``ImageType``
                accepts either ``numpy.ndarray`` or ``PIL.Image.Image``.
            key_points: A collection of key points. Covariance matrices must be
                stored in ``key_points.data[covariance_data_key]``.

        Returns:
            The annotated image, matching the type of ``scene``.

        Raises:
            ValueError: If ``key_points.data`` does not contain the key specified
                by ``covariance_data_key``, or if the covariance array shape does
                not match ``(N, K, 2, 2)``.

        Example:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
            >>> key_points = sv.KeyPoints(
            ...     xy=np.array([[[50, 50], [60, 60]]], dtype=np.float32),
            ...     data={"covariance": np.array([[[[25, 0], [0, 9]], [[9, 0], [0, 4]]]], dtype=np.float32)}
            ... )
            >>> annotator = sv.VertexEllipseAnnotator(color=sv.Color.GREEN)
            >>> annotated_frame = annotator.annotate(image.copy(), key_points)
            >>> annotated_frame.shape
            (100, 100, 3)

            ```
        """  # noqa: E501 // docs
        assert isinstance(scene, np.ndarray)
        if len(key_points) == 0:
            return scene

        covariances = self._get_covariances(key_points=key_points)
        for detection_index, xy in enumerate(key_points.xy):
            for point_index, (x, y) in enumerate(xy):
                if np.allclose((x, y), 0):
                    continue
                if key_points.confidence is not None:
                    confidence = key_points.confidence[detection_index, point_index]
                    if not np.isfinite(confidence):
                        continue
                ellipse = self._covariance_to_ellipse(
                    covariance=covariances[detection_index, point_index]
                )
                if ellipse is None:
                    continue
                axis_lengths, angle = ellipse
                self._draw_ellipse(
                    scene=scene,
                    center=(round(x), round(y)),
                    axes=axis_lengths,
                    angle=angle,
                )

        return scene

    def _get_covariances(self, key_points: KeyPoints) -> npt.NDArray[np.float32]:
        covariances = key_points.data.get(self.covariance_data_key)
        if covariances is None:
            raise ValueError(
                f"key_points.data must contain {self.covariance_data_key!r} "
                "with shape (N, K, 2, 2)."
            )
        covariances_array = cast(
            npt.NDArray[np.float32], np.asarray(covariances, dtype=np.float32)
        )
        expected_shape = (*key_points.xy.shape[:2], 2, 2)
        if covariances_array.shape != expected_shape:
            raise ValueError(
                f"Expected covariance shape {expected_shape}, "
                f"got {covariances_array.shape}."
            )
        return covariances_array

    def _covariance_to_ellipse(
        self, covariance: npt.NDArray[np.float32]
    ) -> tuple[tuple[int, int], float] | None:
        if not np.isfinite(covariance).all():
            return None
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance.astype(np.float64))
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0):
            return None

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        axes = self.sigma * np.sqrt(eigenvalues)
        if self.max_axis_length is not None:
            axes = np.minimum(axes, self.max_axis_length)
        axis_lengths = tuple(max(1, round(axis)) for axis in axes)
        angle = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))
        return axis_lengths, angle

    def _draw_ellipse(
        self,
        scene: npt.NDArray[np.uint8],
        center: tuple[int, int],
        axes: tuple[int, int],
        angle: float,
    ) -> None:
        if self.line_style == "solid":
            cv2.ellipse(
                img=scene,
                center=center,
                axes=axes,
                angle=angle,
                startAngle=0,
                endAngle=360,
                color=self.color.as_bgr(),
                thickness=self.thickness,
                lineType=cv2.LINE_AA,
            )
            return

        step = self.dash_length * 2
        for start_angle in range(0, 360, step):
            cv2.ellipse(
                img=scene,
                center=center,
                axes=axes,
                angle=angle,
                startAngle=start_angle,
                endAngle=min(start_angle + self.dash_length, 360),
                color=self.color.as_bgr(),
                thickness=self.thickness,
                lineType=cv2.LINE_AA,
            )


class VertexLabelAnnotator:
    """
    A class that draws labels of skeleton vertices on images. It uses specified key
    points to determine the locations where the vertices should be drawn.
    """

    def __init__(
        self,
        color: Color | list[Color] = Color.ROBOFLOW,
        text_color: Color | list[Color] = Color.WHITE,
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
        border_radius: int = 0,
        smart_position: bool = False,
    ):
        """
        Args:
            color: The color to use for each keypoint label. If a list is provided,
                the colors will be used in order for each keypoint.
            text_color: The color to use for the labels. If a list is provided, the
                colors will be used in order for each keypoint.
            text_scale: The scale of the text.
            text_thickness: The thickness of the text.
            text_padding: The padding around the text.
            border_radius: The radius of the rounded corners of the boxes. Set to a
                high value to produce circles.
            smart_position: Spread out the labels to avoid overlap.
        """
        self.border_radius: int = border_radius
        self.color: Color | list[Color] = color
        self.text_color: Color | list[Color] = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding
        self.smart_position = smart_position

    def annotate(
        self,
        scene: ImageType,
        key_points: KeyPoints,
        labels: list[str] | None = None,
    ) -> ImageType:
        """
        A class that draws labels of skeleton vertices on images. It uses specified key
            points to determine the locations where the vertices should be drawn.

        Args:
            scene: The image where vertex labels will be drawn. `ImageType` is a
                flexible type, accepting either `numpy.ndarray` or `PIL.Image.Image`.
            key_points: A collection of key points where each key point consists of x
                and y coordinates.
            labels: A list of labels to be displayed on the annotated image. If not
                provided, keypoint indices will be used.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
            >>> key_points = sv.KeyPoints(
            ...     xy=np.array([[[50, 50], [60, 60]]], dtype=np.float32)
            ... )
            >>> vertex_label_annotator = sv.VertexLabelAnnotator(
            ...     color=sv.Color.GREEN,
            ...     text_color=sv.Color.BLACK,
            ...     border_radius=5
            ... )
            >>> annotated_frame = vertex_label_annotator.annotate(
            ...     scene=image.copy(),
            ...     key_points=key_points
            ... )
            >>> annotated_frame.shape
            (100, 100, 3)

            ```

        ![vertex-label-annotator-example](https://media.roboflow.com/supervision-annotator-examples/vertex-label-annotator-example.png)

        !!! tip

            `VertexLabelAnnotator` allows to customize the color of each keypoint label
            values.

        Example:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
            >>> key_points = sv.KeyPoints(
            ...     xy=np.array([[[50, 50], [60, 60], [70, 70], [80, 80],
            ...                   [90, 90], [40, 40], [30, 30], [20, 20],
            ...                   [10, 10], [5, 5], [15, 15], [25, 25],
            ...                   [35, 35], [45, 45], [55, 55], [65, 65],
            ...                   [75, 75]]], dtype=np.float32)
            ... )
            >>> LABELS = [
            ...     "nose", "left eye", "right eye", "left ear",
            ...     "right ear", "left shoulder", "right shoulder", "left elbow",
            ...     "right elbow", "left wrist", "right wrist", "left hip",
            ...     "right hip", "left knee", "right knee", "left ankle",
            ...     "right ankle"
            ... ]
            >>> COLORS = [
            ...     "#FF6347", "#FF6347", "#FF6347", "#FF6347",
            ...     "#FF6347", "#FF1493", "#00FF00", "#FF1493",
            ...     "#00FF00", "#FF1493", "#00FF00", "#FFD700",
            ...     "#00BFFF", "#FFD700", "#00BFFF", "#FFD700",
            ...     "#00BFFF"
            ... ]
            >>> COLORS = [sv.Color.from_hex(color_hex=c) for c in COLORS]
            >>> vertex_label_annotator = sv.VertexLabelAnnotator(
            ...     color=COLORS,
            ...     text_color=sv.Color.BLACK,
            ...     border_radius=5
            ... )
            >>> annotated_frame = vertex_label_annotator.annotate(
            ...     scene=image.copy(),
            ...     key_points=key_points,
            ...     labels=LABELS
            ... )
            >>> annotated_frame.shape
            (100, 100, 3)

            ```

        ![vertex-label-annotator-custom-example](https://media.roboflow.com/supervision-annotator-examples/vertex-label-annotator-custom-example.png)
        """
        assert isinstance(scene, np.ndarray)
        font = cv2.FONT_HERSHEY_SIMPLEX

        skeletons_count, points_count, _ = key_points.xy.shape
        if skeletons_count == 0:
            return scene

        anchors = cast(
            npt.NDArray[np.int_],
            key_points.xy.reshape(points_count * skeletons_count, 2).astype(int),
        )
        mask = np.all(anchors != 0, axis=1)

        if not np.any(mask):
            return scene

        colors = self.preprocess_and_validate_colors(
            colors=self.color,
            points_count=points_count,
            skeletons_count=skeletons_count,
        )

        text_colors = self.preprocess_and_validate_colors(
            colors=self.text_color,
            points_count=points_count,
            skeletons_count=skeletons_count,
        )

        processed_labels = self.preprocess_and_validate_labels(
            labels=labels, points_count=points_count, skeletons_count=skeletons_count
        )

        anchors = anchors[mask]
        colors = colors[mask]
        text_colors = text_colors[mask]
        filtered_labels = processed_labels[mask]

        xyxy = np.array(
            [
                self.get_text_bounding_box(
                    text=label,
                    font=font,
                    text_scale=self.text_scale,
                    text_thickness=self.text_thickness,
                    center_coordinates=tuple(anchor),
                )
                for anchor, label in zip(anchors, filtered_labels)
            ]
        )
        xyxy_padded = pad_boxes(xyxy=xyxy, px=self.text_padding)

        if self.smart_position:
            xyxy_padded = spread_out_boxes(xyxy_padded)
            xyxy = pad_boxes(xyxy=xyxy_padded, px=-self.text_padding)

        for text, color, text_color, box, box_padded in zip(
            filtered_labels, colors, text_colors, xyxy, xyxy_padded
        ):
            draw_rounded_rectangle(
                scene=scene,
                rect=Rect.from_xyxy(box_padded),
                color=color,
                border_radius=self.border_radius,
            )
            cv2.putText(
                img=scene,
                text=text,
                org=(box[0], box[3]),
                fontFace=font,
                fontScale=self.text_scale,
                color=text_color.as_bgr(),
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )

        return scene

    @staticmethod
    def get_text_bounding_box(
        text: str,
        font: int,
        text_scale: float,
        text_thickness: int,
        center_coordinates: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        text_w, text_h = cv2.getTextSize(
            text=text,
            fontFace=font,
            fontScale=text_scale,
            thickness=text_thickness,
        )[0]
        center_x, center_y = center_coordinates
        return (
            center_x - text_w // 2,
            center_y - text_h // 2,
            center_x + text_w // 2,
            center_y + text_h // 2,
        )

    @staticmethod
    def preprocess_and_validate_labels(
        labels: list[str] | None, points_count: int, skeletons_count: int
    ) -> npt.NDArray[np.str_]:
        if labels and len(labels) != points_count:
            raise ValueError(
                f"Number of labels ({len(labels)}) must match number of key points "
                f"({points_count})."
            )
        if labels is None:
            labels = [str(i) for i in range(points_count)]

        return np.array(labels * skeletons_count)

    @staticmethod
    def preprocess_and_validate_colors(
        colors: Color | list[Color] | None,
        points_count: int,
        skeletons_count: int,
    ) -> npt.NDArray[Any]:
        if isinstance(colors, list) and len(colors) != points_count:
            raise ValueError(
                f"Number of colors ({len(colors)}) must match number of key points "
                f"({points_count})."
            )
        return (
            np.array(colors * skeletons_count)
            if isinstance(colors, list)
            else np.array([colors] * points_count * skeletons_count)
        )
