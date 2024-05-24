from abc import ABC, abstractmethod
from logging import warn
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from supervision import Rect, pad_boxes
from supervision.annotators.base import ImageType
from supervision.draw.color import Color
from supervision.draw.utils import draw_rounded_rectangle
from supervision.keypoint.core import KeyPoints
from supervision.keypoint.skeletons import SKELETONS_BY_VERTEX_COUNT
from supervision.utils.conversion import convert_for_annotation_method


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
            color (Color, optional): The color to use for annotating key points.
            radius (int, optional): The radius of the circles used to represent the key
                points.
        """
        self.color = color
        self.radius = radius

    @convert_for_annotation_method
    def annotate(self, scene: ImageType, key_points: KeyPoints) -> ImageType:
        """
        Annotates the given scene with skeleton vertices based on the provided key
        points. It draws circles at each key point location.

        Args:
            scene (ImageType): The image where skeleton vertices will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray` or
                `PIL.Image.Image`.
            key_points (KeyPoints): A collection of key points where each key point
                consists of x and y coordinates.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            key_points = sv.KeyPoints(...)

            vertex_annotator = sv.VertexAnnotator(
                color=sv.Color.GREEN,
                radius=10
            )
            annotated_frame = vertex_annotator.annotate(
                scene=image.copy(),
                key_points=key_points
            )
            ```

        ![vertex-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/vertex-annotator-example.png)
        """
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
        edges: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        """
        Args:
            color (Color, optional): The color to use for the edges.
            thickness (int, optional): The thickness of the edges.
            edges (Optional[List[Tuple[int, int]]]): The edges to draw.
                If set to `None`, will attempt to select automatically.
        """
        self.color = color
        self.thickness = thickness
        self.edges = edges

    @convert_for_annotation_method
    def annotate(self, scene: ImageType, key_points: KeyPoints) -> ImageType:
        """
        Annotates the given scene by drawing lines between specified key points to form
        edges.

        Args:
            scene (ImageType): The image where skeleton edges will be drawn. `ImageType`
                is a flexible type, accepting either `numpy.ndarray` or
                `PIL.Image.Image`.
            key_points (KeyPoints): A collection of key points where each key point
                consists of x and y coordinates.

        Returns:
            Returns:
                The annotated image, matching the type of `scene` (`numpy.ndarray`
                    or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            key_points = sv.KeyPoints(...)

            edge_annotator = sv.EdgeAnnotator(
                color=sv.Color.GREEN,
                thickness=5
            )
            annotated_frame = edge_annotator.annotate(
                scene=image.copy(),
                key_points=key_points
            )
            ```

        ![edge-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/edge-annotator-example.png)
        """
        if len(key_points) == 0:
            return scene

        for xy in key_points.xy:
            edges = self.edges
            if not edges:
                edges = SKELETONS_BY_VERTEX_COUNT.get(len(xy))
            if not edges:
                warn(f"No skeleton found with {len(xy)} vertices")
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


class VertexLabelAnnotator:
    """
    A class that draws labels of skeleton vertices on images. It uses specified key
    points to determine the locations where the vertices should be drawn.
    """

    def __init__(
        self,
        color: Union[Color, List[Color]] = Color.ROBOFLOW,
        text_color: Color = Color.WHITE,
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
        border_radius: int = 0,
    ):
        """
        Args:
            color (Union[Color, List[Color]], optional): The color to use for each
                keypoint label. If a list is provided, the colors will be used in order
                for each keypoint.
            text_color (Color, optional): The color to use for the labels.
            text_scale (float, optional): The scale of the text.
            text_thickness (int, optional): The thickness of the text.
            text_padding (int, optional): The padding around the text.
            border_radius (int, optional): The radius of the rounded corners of the
                boxes. Set to a high value to produce circles.
        """
        self.border_radius: int = border_radius
        self.color: Union[Color, List[Color]] = color
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding

    def annotate(
        self, scene: ImageType, key_points: KeyPoints, labels: List[str] = None
    ) -> ImageType:
        """
        A class that draws labels of skeleton vertices on images. It uses specified key
            points to determine the locations where the vertices should be drawn.

        Args:
            scene (ImageType): The image where vertex labels will be drawn. `ImageType`
                is a flexible type, accepting either `numpy.ndarray` or
                `PIL.Image.Image`.
            key_points (KeyPoints): A collection of key points where each key point
                consists of x and y coordinates.
            labels (List[str], optional): A list of labels to be displayed on the
                annotated image. If not provided, keypoint indices will be used.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            key_points = sv.KeyPoints(...)

            vertex_label_annotator = sv.VertexLabelAnnotator(
                color=sv.Color.GREEN,
                text_color=sv.Color.BLACK,
                border_radius=5
            )
            annotated_frame = vertex_label_annotator.annotate(
                scene=image.copy(),
                key_points=key_points
            )
            ```

        ![vertex-label-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/vertex-label-annotator-example.png)

        !!! tip

            `VertexLabelAnnotator` allows to customize the color of each keypoint label
            values.

        Example:
            ```python
            import supervision as sv

            image = ...
            key_points = sv.KeyPoints(...)

            LABELS = [
                "nose", "left eye", "right eye", "left ear",
                "right ear", "left shoulder", "right shoulder", "left elbow",
                "right elbow", "left wrist", "right wrist", "left hip",
                "right hip", "left knee", "right knee", "left ankle",
                "right ankle"
            ]

            COLORS = [
                "#FF6347", "#FF6347", "#FF6347", "#FF6347",
                "#FF6347", "#FF1493", "#00FF00", "#FF1493",
                "#00FF00", "#FF1493", "#00FF00", "#FFD700",
                "#00BFFF", "#FFD700", "#00BFFF", "#FFD700",
                "#00BFFF"
            ]
            COLORS = [sv.Color.from_hex(color_hex=c) for c in COLORS]

            vertex_label_annotator = sv.VertexLabelAnnotator(
                color=COLORS,
                text_color=sv.Color.BLACK,
                border_radius=5
            )
            annotated_frame = vertex_label_annotator.annotate(
                scene=image.copy(),
                key_points=key_points,
                labels=labels
            )
            ```
        ![vertex-label-annotator-custom-example](https://media.roboflow.com/
        supervision-annotator-examples/vertex-label-annotator-custom-example.png)
        """
        font = cv2.FONT_HERSHEY_SIMPLEX

        skeletons_count, points_count, _ = key_points.xy.shape
        if skeletons_count == 0:
            return scene

        anchors = key_points.xy.reshape(points_count * skeletons_count, 2).astype(int)
        mask = np.all(anchors != 0, axis=1)

        if not np.any(mask):
            return scene

        colors = self.preprocess_and_validate_colors(
            colors=self.color,
            points_count=points_count,
            skeletons_count=skeletons_count,
        )

        labels = self.preprocess_and_validate_labels(
            labels=labels, points_count=points_count, skeletons_count=skeletons_count
        )

        anchors = anchors[mask]
        colors = colors[mask]
        labels = labels[mask]

        xyxy = np.array(
            [
                self.get_text_bounding_box(
                    text=label,
                    font=font,
                    text_scale=self.text_scale,
                    text_thickness=self.text_thickness,
                    center_coordinates=tuple(anchor),
                )
                for anchor, label in zip(anchors, labels)
            ]
        )

        xyxy_padded = pad_boxes(xyxy=xyxy, px=self.text_padding)

        for text, color, box, box_padded in zip(labels, colors, xyxy, xyxy_padded):
            draw_rounded_rectangle(
                scene=scene,
                rect=Rect.from_xyxy(box_padded),
                color=color,
                border_radius=self.border_radius,
            )
            cv2.putText(
                img=scene,
                text=text,
                org=(box[0], box[1] + self.text_padding),
                fontFace=font,
                fontScale=self.text_scale,
                color=self.text_color.as_rgb(),
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
        center_coordinates: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
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
        labels: Optional[List[str]], points_count: int, skeletons_count: int
    ) -> np.array:
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
        colors: Optional[Union[Color, List[Color]]],
        points_count: int,
        skeletons_count: int,
    ) -> np.array:
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
