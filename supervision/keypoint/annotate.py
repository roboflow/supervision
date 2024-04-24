from abc import ABC, abstractmethod
from logging import warn
from typing import List, Optional, Tuple

import cv2
import numpy as np

from supervision.annotators.base import ImageType
from supervision.draw.color import Color
from supervision.keypoint.core import KeyPoints
from supervision.keypoint.skeletons import resolve_skeleton_by_vertex_count
from supervision.utils.conversion import convert_for_annotation_method


class BaseKeyPointAnnotator(ABC):
    @abstractmethod
    def annotate(self, scene: ImageType, detections: KeyPoints) -> ImageType:
        pass


class VertexAnnotator(BaseKeyPointAnnotator):
    def __init__(
        self,
        color: Color = Color.ROBOFLOW,
        radius: int = 4,
    ) -> None:
        """
        Most basic keypoint annotator.

        Args:
            color (Color, optional): The color of the keypoint.
            radius (int, optional): The radius of the keypoint.
        """
        self.color = color
        self.radius = radius

    @convert_for_annotation_method
    def annotate(self, scene: ImageType, keypoints: KeyPoints) -> ImageType:
        if len(keypoints) == 0:
            return scene

        for xy in keypoints.xy:
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
    def __init__(
        self,
        color: Color = Color.ROBOFLOW,
        thickness: int = 2,
        edges: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        """
        Draw the lines between points of the image.

        Args:
            color (Color, optional): The color of the lines.
            thickness (int, optional): The thickness of the lines.
            edge (Optional[List[Tuple[int, int]]]): The edges to draw.
                If set to `None`, will attempt to select automatically.
        """
        self.color = color
        self.thickness = thickness
        self.edges = edges

    @convert_for_annotation_method
    def annotate(self, scene: ImageType, keypoints: KeyPoints) -> ImageType:
        if len(keypoints) == 0:
            return scene

        for xy in keypoints.xy:
            edges = self.edges
            if not edges:
                edges = resolve_skeleton_by_vertex_count(len(xy))
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
