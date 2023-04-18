from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from supervision.detection.core import Detections
from supervision.dataset.formats.pascal_voc import dataset_to_pascal_voc


@dataclass
class Dataset:
    """
    Dataclass containing information about the dataset.

    Attributes:
        classes (List[str]): List containing dataset class names.
        images (Dict[str, np.ndarray]): Dictionary mapping image path to image.
        annotations (Dict[str, Detections]): Dictionary mapping image path to annotations.
    """
    classes: List[str]
    images: Dict[str, np.ndarray]
    annotations: Dict[str, Detections]

    def as_pascal_voc(
            self,
            images_directory_path: Optional[str],
            annotations_directory_path: Optional[str]
    ) -> None:
        dataset_to_pascal_voc(
            detections=self,
            images_directory_path=images_directory_path,
            annotations_directory_path=annotations_directory_path
        )
