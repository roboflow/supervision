from abc import ABC, abstractmethod
from logging import warn
from typing import List, Optional, Tuple

import cv2
import numpy as np

from supervision.annotators.base import ImageType
from supervision.draw.color import Color
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
            scene (ImageType): The image where bounding boxes will be drawn. `ImageType`
                is a flexible type, accepting either `numpy.ndarray` or
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

            vertex_annotator = sv.VertexAnnotator(color=sv.Color.GREEN, radius=10)
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
            scene (ImageType): The image where bounding boxes will be drawn. `ImageType`
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

            edge_annotator = sv.EdgeAnnotator(color=sv.Color.GREEN, thickness=5)
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
