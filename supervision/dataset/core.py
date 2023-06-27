from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from supervision.classification.core import Classifications
from supervision.dataset.formats.coco import (
    load_coco_annotations,
    save_coco_annotations,
)
from supervision.dataset.formats.pascal_voc import (
    detections_to_pascal_voc,
    load_pascal_voc_annotations,
)
from supervision.dataset.formats.yolo import (
    load_yolo_annotations,
    save_data_yaml,
    save_yolo_annotations,
)
from supervision.dataset.ultils import save_dataset_images, train_test_split
from supervision.detection.core import Detections
from supervision.utils.file import list_files_with_extensions


@dataclass
class BaseDataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def split(
        self, split_ratio=0.8, random_state=None, shuffle: bool = True
    ) -> Tuple[BaseDataset, BaseDataset]:
        pass


@dataclass
class DetectionDataset(BaseDataset):
    """
    Dataclass containing information about object detection dataset.

    Attributes:
        classes (List[str]): List containing dataset class names.
        images (Dict[str, np.ndarray]): Dictionary mapping image name to image.
        annotations (Dict[str, Detections]): Dictionary mapping image name to annotations.
    """

    classes: List[str]
    images: Dict[str, np.ndarray]
    annotations: Dict[str, Detections]

    def __len__(self) -> int:
        """
        Return the number of images in the dataset.

        Returns:
            int: The number of images.
        """
        return len(self.images)

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray, Detections]]:
        """
        Iterate over the images and annotations in the dataset.

        Yields:
            Iterator[Tuple[str, np.ndarray, Detections]]: An iterator that yields tuples containing the image name,
                                                          the image data, and its corresponding annotation.
        """
        for image_name, image in self.images.items():
            yield image_name, image, self.annotations.get(image_name, None)

    def split(
        self, split_ratio=0.8, random_state=None, shuffle: bool = True
    ) -> Tuple[DetectionDataset, DetectionDataset]:
        """
        Splits the dataset into two parts (training and testing) using the provided split_ratio.

        Args:
            split_ratio (float, optional): The ratio of the training set to the entire dataset.
            random_state (int, optional): The seed for the random number generator. This is used for reproducibility.
            shuffle (bool, optional): Whether to shuffle the data before splitting.

        Returns:
            Tuple[DetectionDataset, DetectionDataset]: A tuple containing the training and testing datasets.

        Example:
            ```python
            >>> import supervision as sv

            >>> ds = sv.DetectionDataset(...)
            >>> train_ds, test_ds = ds.split(split_ratio=0.7, random_state=42, shuffle=True)
            >>> len(train_ds), len(test_ds)
            (700, 300)
            ```
        """

        image_names = list(self.images.keys())
        train_names, test_names = train_test_split(
            data=image_names,
            train_ratio=split_ratio,
            random_state=random_state,
            shuffle=shuffle,
        )

        train_dataset = DetectionDataset(
            classes=self.classes,
            images={name: self.images[name] for name in train_names},
            annotations={name: self.annotations[name] for name in train_names},
        )
        test_dataset = DetectionDataset(
            classes=self.classes,
            images={name: self.images[name] for name in test_names},
            annotations={name: self.annotations[name] for name in test_names},
        )
        return train_dataset, test_dataset

    def as_pascal_voc(
        self,
        images_directory_path: Optional[str] = None,
        annotations_directory_path: Optional[str] = None,
        min_image_area_percentage: float = 0.0,
        max_image_area_percentage: float = 1.0,
        approximation_percentage: float = 0.0,
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
    ) -> DetectionDataset:
        """
        Creates a Dataset instance from PASCAL VOC formatted data.

        Args:
            images_directory_path (str): The path to the directory containing the images.
            annotations_directory_path (str): The path to the directory containing the PASCAL VOC XML annotations.

        Returns:
            DetectionDataset: A DetectionDataset instance containing the loaded images and annotations.

        Example:
            ```python
            >>> import roboflow
            >>> from roboflow import Roboflow
            >>> import supervision as sv

            >>> roboflow.login()

            >>> rf = Roboflow()

            >>> project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
            >>> dataset = project.version(PROJECT_VERSION).download("voc")

            >>> ds = sv.DetectionDataset.from_yolo(
            ...     images_directory_path=f"{dataset.location}/train/images",
            ...     annotations_directory_path=f"{dataset.location}/train/labels"
            ... )

            >>> ds.classes
            ['dog', 'person']
            ```
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
        return DetectionDataset(classes=classes, images=images, annotations=annotations)

    @classmethod
    def from_yolo(
        cls,
        images_directory_path: str,
        annotations_directory_path: str,
        data_yaml_path: str,
        force_masks: bool = False,
    ) -> DetectionDataset:
        """
        Creates a Dataset instance from YOLO formatted data.

        Args:
            images_directory_path (str): The path to the directory containing the images.
            annotations_directory_path (str): The path to the directory containing the YOLO annotation files.
            data_yaml_path (str): The path to the data YAML file containing class information.
            force_masks (bool, optional): If True, forces masks to be loaded for all annotations, regardless of whether they are present.

        Returns:
            DetectionDataset: A DetectionDataset instance containing the loaded images and annotations.

        Example:
            ```python
            >>> import roboflow
            >>> from roboflow import Roboflow
            >>> import supervision as sv

            >>> roboflow.login()

            >>> rf = Roboflow()

            >>> project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
            >>> dataset = project.version(PROJECT_VERSION).download("yolov5")

            >>> ds = sv.DetectionDataset.from_yolo(
            ...     images_directory_path=f"{dataset.location}/train/images",
            ...     annotations_directory_path=f"{dataset.location}/train/labels",
            ...     data_yaml_path=f"{dataset.location}/data.yaml"
            ... )

            >>> ds.classes
            ['dog', 'person']
            ```
        """
        classes, images, annotations = load_yolo_annotations(
            images_directory_path=images_directory_path,
            annotations_directory_path=annotations_directory_path,
            data_yaml_path=data_yaml_path,
            force_masks=force_masks,
        )
        return DetectionDataset(classes=classes, images=images, annotations=annotations)

    def as_yolo(
        self,
        images_directory_path: Optional[str] = None,
        annotations_directory_path: Optional[str] = None,
        data_yaml_path: Optional[str] = None,
        min_image_area_percentage: float = 0.0,
        max_image_area_percentage: float = 1.0,
        approximation_percentage: float = 0.0,
    ) -> None:
        """
        Exports the dataset to YOLO format. This method saves the images and their corresponding
        annotations in YOLO format, which is a simple text file that describes an object in the image. It also allows
        for the optional saving of a data.yaml file, used in YOLOv5, that contains metadata about the dataset.

        The method allows filtering the detections based on their area percentage and offers an option for polygon approximation.

        Args:
            images_directory_path (Optional[str]): The path to the directory where the images should be saved.
                If not provided, images will not be saved.
            annotations_directory_path (Optional[str]): The path to the directory where the annotations in
                YOLO format should be saved. If not provided, annotations will not be saved.
            data_yaml_path (Optional[str]): The path where the data.yaml file should be saved.
                If not provided, the file will not be saved.
            min_image_area_percentage (float): The minimum percentage of detection area relative to
                the image area for a detection to be included.
            max_image_area_percentage (float): The maximum percentage of detection area relative to
                the image area for a detection to be included.
            approximation_percentage (float): The percentage of polygon points to be removed from the input polygon,
                in the range [0, 1). This is useful for simplifying the annotations.
        """
        if images_directory_path is not None:
            save_dataset_images(
                images_directory_path=images_directory_path, images=self.images
            )
        if annotations_directory_path is not None:
            save_yolo_annotations(
                annotations_directory_path=annotations_directory_path,
                images=self.images,
                annotations=self.annotations,
                min_image_area_percentage=min_image_area_percentage,
                max_image_area_percentage=max_image_area_percentage,
                approximation_percentage=approximation_percentage,
            )
        if data_yaml_path is not None:
            save_data_yaml(data_yaml_path=data_yaml_path, classes=self.classes)

    @classmethod
    def from_coco(
        cls,
        images_directory_path: str,
        annotations_path: str,
        force_masks: bool = False,
    ) -> DetectionDataset:
        """
        Creates a Dataset instance from YOLO formatted data.

        Args:
            images_directory_path (str): The path to the directory containing the images.
            annotations_path (str): The path to the json annotation files.
            force_masks (bool, optional): If True, forces masks to be loaded for all annotations, regardless of whether they are present.

        Returns:
            DetectionDataset: A DetectionDataset instance containing the loaded images and annotations.

        Example:
            ```python
            >>> import roboflow
            >>> from roboflow import Roboflow
            >>> import supervision as sv

            >>> roboflow.login()

            >>> rf = Roboflow()

            >>> project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
            >>> dataset = project.version(PROJECT_VERSION).download("coco")

            >>> ds = sv.DetectionDataset.from_coco(
            ...     images_directory_path=f"{dataset.location}/train",
            ...     annotations_path=f"{dataset.location}/train/_annotations.coco.json",
            ... )

            >>> ds.classes
            ['dog', 'person']
            ```
        """
        classes, images, annotations = load_coco_annotations(
            images_directory_path=images_directory_path,
            annotations_path=annotations_path,
            force_masks=force_masks,
        )
        return DetectionDataset(classes=classes, images=images, annotations=annotations)

    def as_coco(
        self,
        images_directory_path: Optional[str] = None,
        annotations_path: Optional[str] = None,
        min_image_area_percentage: float = 0.0,
        max_image_area_percentage: float = 1.0,
        approximation_percentage: float = 0.0,
        licenses: Optional[list] = None,
        info: Optional[dict] = None,
    ) -> None:
        """
        Exports the dataset to COCO format. This method saves the images and their corresponding
        annotations in COCO format, which is a simple json file that describes an object in the image.
        Annotation json file also include category maps.

        The method allows filtering the detections based on their area percentage and offers an option for polygon approximation.

        Args:
            images_directory_path (Optional[str]): The path to the directory where the images should be saved.
                If not provided, images will not be saved.
            annotations_directory_path (Optional[str]): The path to the directory where the annotations in
                YOLO format should be saved. If not provided, annotations will not be saved.
            min_image_area_percentage (float): The minimum percentage of detection area relative to
                the image area for a detection to be included.
            max_image_area_percentage (float): The maximum percentage of detection area relative to
                the image area for a detection to be included.
            approximation_percentage (float): The percentage of polygon points to be removed from the input polygon,
                in the range [0, 1). This is useful for simplifying the annotations.
            licenses (Optional[str]): List of licenses for images
            info (Optional[dict]): Information of Dataset as dictionary
        """
        if images_directory_path is not None:
            save_dataset_images(
                images_directory_path=images_directory_path, images=self.images
            )
        if annotations_path is not None:
            save_coco_annotations(
                annotation_path=annotations_path,
                images=self.images,
                annotations=self.annotations,
                classes=self.classes,
                min_image_area_percentage=min_image_area_percentage,
                max_image_area_percentage=max_image_area_percentage,
                approximation_percentage=approximation_percentage,
                licenses=licenses,
                info=info,
            )


@dataclass
class ClassificationDataset(BaseDataset):
    """
    Dataclass containing information about a classification dataset.

    Attributes:
        classes (List[str]): List containing dataset class names.
        images (Dict[str, np.ndarray]): Dictionary mapping image name to image.
        annotations (Dict[str, Detections]): Dictionary mapping image name to annotations.
    """

    classes: List[str]
    images: Dict[str, np.ndarray]
    annotations: Dict[str, Classifications]

    def __len__(self) -> int:
        return len(self.images)

    def split(
        self, split_ratio=0.8, random_state=None, shuffle: bool = True
    ) -> Tuple[ClassificationDataset, ClassificationDataset]:
        """
        Splits the dataset into two parts (training and testing) using the provided split_ratio.

        Args:
            split_ratio (float, optional): The ratio of the training set to the entire dataset.
            random_state (int, optional): The seed for the random number generator. This is used for reproducibility.
            shuffle (bool, optional): Whether to shuffle the data before splitting.

        Returns:
            Tuple[ClassificationDataset, ClassificationDataset]: A tuple containing the training and testing datasets.

        Example:
            ```python
            >>> import supervision as sv

            >>> cd = sv.ClassificationDataset(...)
            >>> train_cd, test_cd = cd.split(split_ratio=0.7, random_state=42, shuffle=True)
            >>> len(train_cd), len(test_cd)
            (700, 300)
            ```
        """
        image_names = list(self.images.keys())
        train_names, test_names = train_test_split(
            data=image_names,
            train_ratio=split_ratio,
            random_state=random_state,
            shuffle=shuffle,
        )

        train_dataset = ClassificationDataset(
            classes=self.classes,
            images={name: self.images[name] for name in train_names},
            annotations={name: self.annotations[name] for name in train_names},
        )
        test_dataset = ClassificationDataset(
            classes=self.classes,
            images={name: self.images[name] for name in test_names},
            annotations={name: self.annotations[name] for name in test_names},
        )
        return train_dataset, test_dataset

    def as_folder_structure(self, root_directory_path: str) -> None:
        """
        Saves the dataset as a multi-class folder structure.

        Args:
            root_directory_path (str): The path to the directory where the dataset will be saved.
        """
        os.makedirs(root_directory_path, exist_ok=True)

        for class_name in self.classes:
            os.makedirs(os.path.join(root_directory_path, class_name), exist_ok=True)

        for image_name in self.images:
            classification = self.annotations[image_name]
            image = self.images[image_name]
            class_id = (
                classification.class_id[0]
                if classification.confidence is None
                else classification.get_top_k(1)[0]
            )
            class_name = self.classes[class_id]
            image_path = os.path.join(root_directory_path, class_name, image_name)
            cv2.imwrite(image_path, image)

    @classmethod
    def from_folder_structure(cls, root_directory_path: str) -> ClassificationDataset:
        """
        Load data from a multiclass folder structure into a ClassificationDataset.

        Args:
            root_directory_path (str): The path to the dataset directory.

        Returns:
            ClassificationDataset: The dataset.

        Example:
            ```python
            >>> import roboflow
            >>> from roboflow import Roboflow
            >>> import supervision as sv

            >>> roboflow.login()

            >>> rf = Roboflow()

            >>> project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
            >>> dataset = project.version(PROJECT_VERSION).download("folder")

            >>> cd = sv.ClassificationDataset.from_folder_structure(
            ...     root_directory_path=f"{dataset.location}/train"
            ... )
            ```
        """
        classes = os.listdir(root_directory_path)
        classes = sorted(set(classes))

        images = {}
        annotations = {}

        for class_name in classes:
            class_id = classes.index(class_name)

            for image in os.listdir(os.path.join(root_directory_path, class_name)):
                image_dir = os.path.join(root_directory_path, class_name, image)
                images[image] = cv2.imread(image_dir)
                annotations[image] = Classifications(
                    class_id=np.array([class_id]),
                )

        return cls(
            classes=classes,
            images=images,
            annotations=annotations,
        )
