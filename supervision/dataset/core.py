from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from supervision.dataset.formats.pascal_voc import (
    detections_to_pascal_voc,
    load_pascal_voc_annotations,
)
from supervision.detection.core import Detections
from supervision.file import list_files_with_extensions


@dataclass
class Dataset:
    """
    Dataclass containing information about the dataset.

    Attributes:
        classes (List[str]): List containing dataset class names.
        images (Dict[str, np.ndarray]): Dictionary mapping image name to image.
        annotations (Dict[str, Detections]): Dictionary mapping image name to annotations.
    """

    classes: List[str]
    images: Dict[str, np.ndarray]
    annotations: Dict[str, Detections]

    def as_pascal_voc(
        self,
        images_directory_path: Optional[str] = None,
        annotations_directory_path: Optional[str] = None,
        min_image_area_percentage: float = 0.0,
        max_image_area_percentage: float = 1.0,
        approximation_percentage: float = 0.75,
    ) -> None:
        """
        Exports the dataset to PASCAL VOC format. This method saves the images and their corresponding annotations in
        PASCAL VOC format, which consists of XML files. The method allows filtering the detections based on their area
        percentage.

        Args:
            images_directory_path (Optional[str]): The path to the directory where the images should be saved.
                If not provided, images will not be saved.
            annotations_directory_path (Optional[str]): The path to the directory where the annotations in
                PASCAL VOC format should be saved. If not provided, annotations will not be saved.
            min_image_area_percentage (float): The minimum percentage of detection area relative to
                the image area for a detection to be included.
            max_image_area_percentage (float): The maximum percentage of detection area relative to
                the image area for a detection to be included.
            approximation_percentage (float): The percentage of polygon points to be removed from the input polygon, in the range [0, 1).
        """
        if images_directory_path:
            images_path = Path(images_directory_path)
            images_path.mkdir(parents=True, exist_ok=True)

        if annotations_directory_path:
            annotations_path = Path(annotations_directory_path)
            annotations_path.mkdir(parents=True, exist_ok=True)

        for image_name, image in self.images.items():
            detections = self.annotations[image_name]

            if images_directory_path:
                cv2.imwrite(str(images_path / image_name), image)

            if annotations_directory_path:
                annotation_name = Path(image_name).stem
                pascal_voc_xml = detections_to_pascal_voc(
                    detections=detections,
                    classes=self.classes,
                    filename=image_name,
                    image_shape=image.shape,
                    min_image_area_percentage=min_image_area_percentage,
                    max_image_area_percentage=max_image_area_percentage,
                    approximation_percentage=approximation_percentage,
                )

                with open(annotations_path / f"{annotation_name}.xml", "w") as f:
                    f.write(pascal_voc_xml)

    @classmethod
    def from_pascal_voc(
        cls, images_directory_path: str, annotations_directory_path: str
    ) -> Dataset:
        """
        Creates a Dataset instance from PASCAL VOC formatted data.

        Args:
            images_directory_path (str): The path to the directory containing the images.
            annotations_directory_path (str): The path to the directory containing the PASCAL VOC XML annotations.

        Returns:
            Dataset: A Dataset instance containing the loaded images and annotations.
        """
        image_paths = list_files_with_extensions(
            directory=images_directory_path, extensions=["jpg", "jpeg", "png"]
        )
        annotation_paths = list_files_with_extensions(
            directory=annotations_directory_path, extensions=["xml"]
        )

        raw_annotations: List[Tuple[str, Detections, List[str]]] = [
            load_pascal_voc_annotations(annotation_path=str(annotation_path))
            for annotation_path in annotation_paths
        ]

        classes = []
        for annotation in raw_annotations:
            classes.extend(annotation[2])
        classes = list(set(classes))

        for annotation in raw_annotations:
            class_id = [classes.index(class_name) for class_name in annotation[2]]
            annotation[1].class_id = np.array(class_id)

        images = {
            image_path.name: cv2.imread(str(image_path)) for image_path in image_paths
        }

        annotations = {
            image_name: detections for image_name, detections, _ in raw_annotations
        }
        return Dataset(classes=classes, images=images, annotations=annotations)
