from abc import ABC, abstractmethod

import cv2
import numpy as np

from supervision.annotators.base import ImageType
from supervision.annotators.utils import scene_to_annotator_img_type
from supervision.draw.color import Color
from supervision.keypoints.core import KeyPoints, Skeleton


class BaseKeyPointAnnotator(ABC):
    @abstractmethod
    def annotate(self, scene: ImageType, detections: KeyPoints) -> ImageType:
        pass


class PointAnnotator(BaseKeyPointAnnotator):
    def __init__(
        self,
        color: Color = Color.GREEN,
        radius: int = 4,
    ) -> None:
        """
        Most basic keypoint annotator.
        """
        self.color = color
        self.radius = radius

    @scene_to_annotator_img_type
    def annotate(self, scene: ImageType, keypoints: KeyPoints) -> ImageType:
        # TODO: I'm getting shape [1, N, 2] here - not [N, 2].
        #       Seems like it accounts for more than 1 person in an image.
        if len(keypoints) == 0:
            return scene

        xy = keypoints.xy[0]
        for i, (x, y) in enumerate(xy):
            cv2.circle(
                img=scene,
                center=(int(x), int(y)),
                radius=self.radius,
                color=self.color.as_bgr(),
                thickness=-1,
            )

        return scene


class SkeletonAnnotator(BaseKeyPointAnnotator):
    def __init__(
        self,
        point_color: Color = Color.GREEN,
        point_radius: int = 4,
        limb_color: Color = Color.GREEN,
        limb_thickness: int = 2,
    ) -> None:
        self.point_color = point_color
        self.point_radius = point_radius
        self.limb_color = limb_color
        self.limb_thickness = limb_thickness

    @scene_to_annotator_img_type
    def annotate(
        self, scene: ImageType, keypoints: KeyPoints, skeleton: Skeleton
    ) -> ImageType:
        if len(keypoints) == 0:
            return scene
        if keypoints.class_id is None:
            raise ValueError("KeyPoints must have class_id to annotate a skeleton")

        # print(keypoints)

        xy = keypoints.xy[0]
        class_id = keypoints.class_id[0]

        for i, (x, y) in enumerate(xy):
            cv2.circle(
                img=scene,
                center=(int(x), int(y)),
                radius=self.point_radius,
                color=self.point_color.as_bgr(),
                thickness=-1,
            )

        for class_a, class_b in skeleton.limbs:
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
                color=self.limb_color.as_bgr(),
                thickness=self.limb_thickness,
            )

        return scene
