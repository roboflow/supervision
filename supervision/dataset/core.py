from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

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
from supervision.dataset.utils import (
    build_class_index_mapping,
    map_detections_class_id,
    merge_class_lists,
    save_dataset_images,
    train_test_split,
)
from supervision.detection.core import Detections
from supervision.utils.internal import deprecated, warn_deprecated
from supervision.utils.iterables import find_duplicates


class BaseDataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def split(
        self,
        split_ratio: float = 0.8,
        random_state: Optional[int] = None,
        shuffle: bool = True,
    ) -> Tuple[BaseDataset, BaseDataset]:
        pass


class DetectionDataset(BaseDataset):
    """
    Contains information about a detection dataset. Handles lazy image loading
    and annotation retrieval, dataset splitting, conversions into multiple
    formats.

    Attributes:
        classes (List[str]): List containing dataset class names.
        images (Union[List[str], Dict[str, np.ndarray]]):
            Accepts a list of image paths, or dictionaries of loaded cv2 images
            with paths as keys. If you pass a list of paths, the dataset will
            lazily load images on demand, which is much more memory-efficient.
        annotations (Dict[str, Detections]): Dictionary mapping
            image path to annotations. The dictionary keys match
            match the keys in `images` or entries in the list of
            image paths.
    """

    def __init__(
        self,
        classes: List[str],
        images: Union[List[str], Dict[str, np.ndarray]],
        annotations: Dict[str, Detections],
    ) -> None:
        self.classes = classes

        if set(images) != set(annotations):
            raise ValueError(
                "The keys of the images and annotations dictionaries must match."
            )
        self.annotations = annotations

        # Eliminate duplicates while preserving order
        self.image_paths = list(dict.fromkeys(images))

        self._images_in_memory: Dict[str, np.ndarray] = {}
        if isinstance(images, dict):
            self._images_in_memory = images
            warn_deprecated(
                "Passing a `Dict[str, np.ndarray]` into `DetectionDataset` is "
                "deprecated and will be removed in `supervision-0.26.0`. Use "
                "a list of paths `List[str]` instead."
            )
            # TODO: when supervision-0.26.0 is released, and Dict[str, np.ndarray]
            #       for images is no longer supported, also simplify the rest of
            #       the code. E.g. list(images) is no longer needed, and merge can
            #       be simplified.

    @property
    @deprecated(
        "`DetectionDataset.images` property is deprecated and will be removed in "
        "`supervision-0.26.0`. Iterate with `for path, image, annotation in dataset:` "
        "instead."
    )
    def images(self) -> Dict[str, np.ndarray]:
        """
        Load all images to memory and return them as a dictionary.

        !!! warning

            Only use this when you need all images at once.
            It is much more memory-efficient to initialize dataset with
            image paths and use `for path, image, annotation in dataset:`.
        """
        if self._images_in_memory:
            return self._images_in_memory

        images = {image_path: cv2.imread(image_path) for image_path in self.image_paths}
        return images

    def _get_image(self, image_path: str) -> np.ndarray:
        """Assumes that image is in dataset"""
        if self._images_in_memory:
            return self._images_in_memory[image_path]
        return cv2.imread(image_path)

    def __len__(self) -> int:
        return len(self._images_in_memory) or len(self.image_paths)

    def __getitem__(self, i: int) -> Tuple[str, np.ndarray, Detections]:
        """
        Returns:
            Tuple[str, np.ndarray, Detections]: The image path, image data,
                and its corresponding annotation at index i.
        """
        image_path = self.image_paths[i]
        image = self._get_image(image_path)
        annotation = self.annotations[image_path]
        return image_path, image, annotation

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray, Detections]]:
        """
        Iterate over the images and annotations in the dataset.

        Yields:
            Iterator[Tuple[str, np.ndarray, Detections]]:
                An iterator that yields tuples containing the image path,
                the image data, and its corresponding annotation.
        """
        for i in range(len(self)):
            image_path, image, annotation = self[i]
            yield image_path, image, annotation

    def __eq__(self, other) -> bool:
        if not isinstance(other, DetectionDataset):
            return False

        if set(self.classes) != set(other.classes):
            return False

        if self.image_paths != other.image_paths:
            return False

        if self._images_in_memory or other._images_in_memory:
            if not np.array_equal(
                list(self._images_in_memory.values()),
                list(other._images_in_memory.values()),
            ):
                return False

        if self.annotations != other.annotations:
            return False

        return True

    def split(
        self,
        split_ratio: float = 0.8,
        random_state: Optional[int] = None,
        shuffle: bool = True,
    ) -> Tuple[DetectionDataset, DetectionDataset]:
        """
        Splits the dataset into two parts (training and testing)
            using the provided split_ratio.

        Args:
            split_ratio (float): The ratio of the training
                set to the entire dataset.
            random_state (Optional[int]): The seed for the random number generator.
                This is used for reproducibility.
            shuffle (bool): Whether to shuffle the data before splitting.

        Returns:
            Tuple[DetectionDataset, DetectionDataset]: A tuple containing
                the training and testing datasets.

        Examples:
            ```python
            import supervision as sv

            ds = sv.DetectionDataset(...)
            train_ds, test_ds = ds.split(split_ratio=0.7, random_state=42, shuffle=True)
            len(train_ds), len(test_ds)
            # (700, 300)
            ```
        """

        train_paths, test_paths = train_test_split(
            data=self.image_paths,
            train_ratio=split_ratio,
            random_state=random_state,
            shuffle=shuffle,
        )

        train_input: Union[List[str], Dict[str, np.ndarray]]
        test_input: Union[List[str], Dict[str, np.ndarray]]
        if self._images_in_memory:
            train_input = {path: self._images_in_memory[path] for path in train_paths}
            test_input = {path: self._images_in_memory[path] for path in test_paths}
        else:
            train_input = train_paths
            test_input = test_paths
        train_annotations = {path: self.annotations[path] for path in train_paths}
        test_annotations = {path: self.annotations[path] for path in test_paths}

        train_dataset = DetectionDataset(
            classes=self.classes,
            images=train_input,
            annotations=train_annotations,
        )
        test_dataset = DetectionDataset(
            classes=self.classes,
            images=test_input,
            annotations=test_annotations,
        )
        return train_dataset, test_dataset

    @classmethod
    def merge(cls, dataset_list: List[DetectionDataset]) -> DetectionDataset:
        """
        Merge a list of `DetectionDataset` objects into a single
            `DetectionDataset` object.

        This method takes a list of `DetectionDataset` objects and combines
        their respective fields (`classes`, `images`,
        `annotations`) into a single `DetectionDataset` object.

        Args:
            dataset_list (List[DetectionDataset]): A list of `DetectionDataset`
                objects to merge.

        Returns:
            (DetectionDataset): A single `DetectionDataset` object containing
            the merged data from the input list.

        Examples:
            ```python
            import supervision as sv

            ds_1 = sv.DetectionDataset(...)
            len(ds_1)
            # 100
            ds_1.classes
            # ['dog', 'person']

            ds_2 = sv.DetectionDataset(...)
            len(ds_2)
            # 200
            ds_2.classes
            # ['cat']

            ds_merged = sv.DetectionDataset.merge([ds_1, ds_2])
            len(ds_merged)
            # 300
            ds_merged.classes
            # ['cat', 'dog', 'person']
            ```
        """

        def is_in_memory(dataset: DetectionDataset) -> bool:
            return len(dataset._images_in_memory) > 0 or len(dataset.image_paths) == 0

        def is_lazy(dataset: DetectionDataset) -> bool:
            return len(dataset._images_in_memory) == 0

        all_in_memory = all([is_in_memory(dataset) for dataset in dataset_list])
        all_lazy = all([is_lazy(dataset) for dataset in dataset_list])
        if not all_in_memory and not all_lazy:
            raise ValueError(
                "Merging lazy and in-memory DetectionDatasets is not supported."
            )

        images_in_memory = {}
        for dataset in dataset_list:
            images_in_memory.update(dataset._images_in_memory)

        image_paths = list(
            chain.from_iterable(dataset.image_paths for dataset in dataset_list)
        )
        image_paths_unique = list(dict.fromkeys(image_paths))
        if len(image_paths) != len(image_paths_unique):
            duplicates = find_duplicates(image_paths)
            raise ValueError(
                f"Image paths {duplicates} are not unique across datasets."
            )
        image_paths = image_paths_unique

        classes = merge_class_lists(
            class_lists=[dataset.classes for dataset in dataset_list]
        )

        annotations = {}
        for dataset in dataset_list:
            annotations.update(dataset.annotations)
        for dataset in dataset_list:
            class_index_mapping = build_class_index_mapping(
                source_classes=dataset.classes, target_classes=classes
            )
            for image_path in dataset.image_paths:
                annotations[image_path] = map_detections_class_id(
                    source_to_target_mapping=class_index_mapping,
                    detections=annotations[image_path],
                )

        return cls(
            classes=classes,
            images=images_in_memory or image_paths,
            annotations=annotations,
        )

    def as_pascal_voc(
        self,
        images_directory_path: Optional[str] = None,
        annotations_directory_path: Optional[str] = None,
        min_image_area_percentage: float = 0.0,
        max_image_area_percentage: float = 1.0,
        approximation_percentage: float = 0.0,
    ) -> None:
        """
        Exports the dataset to PASCAL VOC format. This method saves the images
        and their corresponding annotations in PASCAL VOC format.

        Args:
            images_directory_path (Optional[str]): The path to the directory
                where the images should be saved.
                If not provided, images will not be saved.
            annotations_directory_path (Optional[str]): The path to
                the directory where the annotations in PASCAL VOC format should be
                saved. If not provided, annotations will not be saved.
            min_image_area_percentage (float): The minimum percentage of
                detection area relative to
                the image area for a detection to be included.
                Argument is used only for segmentation datasets.
            max_image_area_percentage (float): The maximum percentage
                of detection area relative to
                the image area for a detection to be included.
                Argument is used only for segmentation datasets.
            approximation_percentage (float): The percentage of
                polygon points to be removed from the input polygon,
                in the range [0, 1). Argument is used only for segmentation datasets.
        """
        if images_directory_path:
            save_dataset_images(
                dataset=self,
                images_directory_path=images_directory_path,
            )
        if annotations_directory_path:
            Path(annotations_directory_path).mkdir(parents=True, exist_ok=True)
            for image_path, image, annotations in self:
                annotation_name = Path(image_path).stem
                annotations_path = os.path.join(
                    annotations_directory_path, f"{annotation_name}.xml"
                )
                image_name = Path(image_path).name
                pascal_voc_xml = detections_to_pascal_voc(
                    detections=annotations,
                    classes=self.classes,
                    filename=image_name,
                    image_shape=image.shape,  # type: ignore
                    min_image_area_percentage=min_image_area_percentage,
                    max_image_area_percentage=max_image_area_percentage,
                    approximation_percentage=approximation_percentage,
                )

                with open(annotations_path, "w") as f:
                    f.write(pascal_voc_xml)

    @classmethod
    def from_pascal_voc(
        cls,
        images_directory_path: str,
        annotations_directory_path: str,
        force_masks: bool = False,
    ) -> DetectionDataset:
        """
        Creates a Dataset instance from PASCAL VOC formatted data.

        Args:
            images_directory_path (str): Path to the directory containing the images.
            annotations_directory_path (str): Path to the directory
                containing the PASCAL VOC XML annotations.
            force_masks (bool): If True, forces masks to
                be loaded for all annotations, regardless of whether they are present.

        Returns:
            DetectionDataset: A DetectionDataset instance containing
                the loaded images and annotations.

        Examples:
            ```python
            import roboflow
            from roboflow import Roboflow
            import supervision as sv

            roboflow.login()

            rf = Roboflow()

            project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
            dataset = project.version(PROJECT_VERSION).download("voc")

            ds = sv.DetectionDataset.from_pascal_voc(
                images_directory_path=f"{dataset.location}/train/images",
                annotations_directory_path=f"{dataset.location}/train/labels"
            )

            ds.classes
            # ['dog', 'person']
            ```
        """

        classes, image_paths, annotations = load_pascal_voc_annotations(
            images_directory_path=images_directory_path,
            annotations_directory_path=annotations_directory_path,
            force_masks=force_masks,
        )

        return DetectionDataset(
            classes=classes, images=image_paths, annotations=annotations
        )

    @classmethod
    def from_yolo(
        cls,
        images_directory_path: str,
        annotations_directory_path: str,
        data_yaml_path: str,
        force_masks: bool = False,
        is_obb: bool = False,
    ) -> DetectionDataset:
        """
        Creates a Dataset instance from YOLO formatted data.

        Args:
            images_directory_path (str): The path to the
                directory containing the images.
            annotations_directory_path (str): The path to the directory
                containing the YOLO annotation files.
            data_yaml_path (str): The path to the data
                YAML file containing class information.
            force_masks (bool): If True, forces
                masks to be loaded for all annotations,
                regardless of whether they are present.
            is_obb (bool): If True, loads the annotations in OBB format.
                OBB annotations are defined as `[class_id, x, y, x, y, x, y, x, y]`,
                where pairs of [x, y] are box corners.

        Returns:
            DetectionDataset: A DetectionDataset instance
                containing the loaded images and annotations.

        Examples:
            ```python
            import roboflow
            from roboflow import Roboflow
            import supervision as sv

            roboflow.login()
            rf = Roboflow()

            project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
            dataset = project.version(PROJECT_VERSION).download("yolov5")

            ds = sv.DetectionDataset.from_yolo(
                images_directory_path=f"{dataset.location}/train/images",
                annotations_directory_path=f"{dataset.location}/train/labels",
                data_yaml_path=f"{dataset.location}/data.yaml"
            )

            ds.classes
            # ['dog', 'person']
            ```
        """
        classes, image_paths, annotations = load_yolo_annotations(
            images_directory_path=images_directory_path,
            annotations_directory_path=annotations_directory_path,
            data_yaml_path=data_yaml_path,
            force_masks=force_masks,
            is_obb=is_obb,
        )
        return DetectionDataset(
            classes=classes, images=image_paths, annotations=annotations
        )

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
        Exports the dataset to YOLO format. This method saves the
        images and their corresponding annotations in YOLO format.

        Args:
            images_directory_path (Optional[str]): The path to the
                directory where the images should be saved.
                If not provided, images will not be saved.
            annotations_directory_path (Optional[str]): The path to the
                directory where the annotations in
                YOLO format should be saved. If not provided,
                annotations will not be saved.
            data_yaml_path (Optional[str]): The path where the data.yaml
                file should be saved.
                If not provided, the file will not be saved.
            min_image_area_percentage (float): The minimum percentage of
                detection area relative to
                the image area for a detection to be included.
                Argument is used only for segmentation datasets.
            max_image_area_percentage (float): The maximum percentage
                of detection area relative to
                the image area for a detection to be included.
                Argument is used only for segmentation datasets.
            approximation_percentage (float): The percentage of polygon points to
                be removed from the input polygon, in the range [0, 1).
                This is useful for simplifying the annotations.
                Argument is used only for segmentation datasets.
        """
        if images_directory_path is not None:
            save_dataset_images(
                dataset=self, images_directory_path=images_directory_path
            )
        if annotations_directory_path is not None:
            save_yolo_annotations(
                dataset=self,
                annotations_directory_path=annotations_directory_path,
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
        Creates a Dataset instance from COCO formatted data.

        Args:
            images_directory_path (str): The path to the
                directory containing the images.
            annotations_path (str): The path to the json annotation files.
            force_masks (bool): If True,
                forces masks to be loaded for all annotations,
                regardless of whether they are present.

        Returns:
            DetectionDataset: A DetectionDataset instance containing
                the loaded images and annotations.

        Examples:
            ```python
            import roboflow
            from roboflow import Roboflow
            import supervision as sv

            roboflow.login()
            rf = Roboflow()

            project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
            dataset = project.version(PROJECT_VERSION).download("coco")

            ds = sv.DetectionDataset.from_coco(
                images_directory_path=f"{dataset.location}/train",
                annotations_path=f"{dataset.location}/train/_annotations.coco.json",
            )

            ds.classes
            # ['dog', 'person']
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
    ) -> None:
        """
        Exports the dataset to COCO format. This method saves the
        images and their corresponding annotations in COCO format.

        !!! tip

            The format of the mask is determined automatically based on its structure:

            - If a mask contains multiple disconnected components or holes, it will be
            saved using the Run-Length Encoding (RLE) format for efficient storage and
            processing.
            - If a mask consists of a single, contiguous region without any holes, it
            will be encoded as a polygon, preserving the outline of the object.

            This automatic selection ensures that the masks are stored in the most
            appropriate and space-efficient format, complying with COCO dataset
            standards.

        Args:
            images_directory_path (Optional[str]): The path to the directory
                where the images should be saved.
                If not provided, images will not be saved.
            annotations_path (Optional[str]): The path to COCO annotation file.
            min_image_area_percentage (float): The minimum percentage of
                detection area relative to
                the image area for a detection to be included.
                Argument is used only for segmentation datasets.
            max_image_area_percentage (float): The maximum percentage of
                detection area relative to
                the image area for a detection to be included.
                Argument is used only for segmentation datasets.
            approximation_percentage (float): The percentage of polygon points
                to be removed from the input polygon,
                in the range [0, 1). This is useful for simplifying the annotations.
                Argument is used only for segmentation datasets.
        """
        if images_directory_path is not None:
            save_dataset_images(
                dataset=self, images_directory_path=images_directory_path
            )
        if annotations_path is not None:
            save_coco_annotations(
                dataset=self,
                annotation_path=annotations_path,
                min_image_area_percentage=min_image_area_percentage,
                max_image_area_percentage=max_image_area_percentage,
                approximation_percentage=approximation_percentage,
            )


@dataclass
class ClassificationDataset(BaseDataset):
    """
    Contains information about a classification dataset, handles lazy image
    loading, dataset splitting.

    Attributes:
        classes (List[str]): List containing dataset class names.
        images (Union[List[str], Dict[str, np.ndarray]]):
            List of image paths or dictionary mapping image name to image data.
        annotations (Dict[str, Classifications]): Dictionary mapping
            image name to annotations.
    """

    def __init__(
        self,
        classes: List[str],
        images: Union[List[str], Dict[str, np.ndarray]],
        annotations: Dict[str, Classifications],
    ) -> None:
        self.classes = classes

        if set(images) != set(annotations):
            raise ValueError(
                "The keys of the images and annotations dictionaries must match."
            )
        self.annotations = annotations

        # Eliminate duplicates while preserving order
        self.image_paths = list(dict.fromkeys(images))

        self._images_in_memory: Dict[str, np.ndarray] = {}
        if isinstance(images, dict):
            self._images_in_memory = images
            warn_deprecated(
                "Passing a `Dict[str, np.ndarray]` into `ClassificationDataset` is "
                "deprecated and will be removed in a future release. Use "
                "a list of paths `List[str]` instead."
            )

    @property
    @deprecated(
        "`DetectionDataset.images` property is deprecated and will be removed in "
        "`supervision-0.26.0`. Iterate with `for path, image, annotation in dataset:` "
        "instead."
    )
    def images(self) -> Dict[str, np.ndarray]:
        """
        Load all images to memory and return them as a dictionary.

        !!! warning

            Only use this when you need all images at once.
            It is much more memory-efficient to initialize dataset with
            image paths and use `for path, image, annotation in dataset:`.
        """
        if self._images_in_memory:
            return self._images_in_memory

        images = {image_path: cv2.imread(image_path) for image_path in self.image_paths}
        return images

    def _get_image(self, image_path: str) -> np.ndarray:
        """Assumes that image is in dataset"""
        if self._images_in_memory:
            return self._images_in_memory[image_path]
        return cv2.imread(image_path)

    def __len__(self) -> int:
        return len(self._images_in_memory) or len(self.image_paths)

    def __getitem__(self, i: int) -> Tuple[str, np.ndarray, Classifications]:
        """
        Returns:
            Tuple[str, np.ndarray, Classifications]: The image path, image data,
                and its corresponding annotation at index i.
        """
        image_path = self.image_paths[i]
        image = self._get_image(image_path)
        annotation = self.annotations[image_path]
        return image_path, image, annotation

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray, Classifications]]:
        """
        Iterate over the images and annotations in the dataset.

        Yields:
            Iterator[Tuple[str, np.ndarray, Detections]]:
                An iterator that yields tuples containing the image path,
                the image data, and its corresponding annotation.
        """
        for i in range(len(self)):
            image_path, image, annotation = self[i]
            yield image_path, image, annotation

    def __eq__(self, other) -> bool:
        if not isinstance(other, ClassificationDataset):
            return False

        if set(self.classes) != set(other.classes):
            return False

        if self.image_paths != other.image_paths:
            return False

        if self._images_in_memory or other._images_in_memory:
            if not np.array_equal(
                list(self._images_in_memory.values()),
                list(other._images_in_memory.values()),
            ):
                return False

        if self.annotations != other.annotations:
            return False

        return True

    def split(
        self,
        split_ratio: float = 0.8,
        random_state: Optional[int] = None,
        shuffle: bool = True,
    ) -> Tuple[ClassificationDataset, ClassificationDataset]:
        """
        Splits the dataset into two parts (training and testing)
            using the provided split_ratio.

        Args:
            split_ratio (float): The ratio of the training
                set to the entire dataset.
            random_state (Optional[int]): The seed for the
                random number generator. This is used for reproducibility.
            shuffle (bool): Whether to shuffle the data before splitting.

        Returns:
            Tuple[ClassificationDataset, ClassificationDataset]: A tuple containing
            the training and testing datasets.

        Examples:
            ```python
            import supervision as sv

            cd = sv.ClassificationDataset(...)
            train_cd,test_cd = cd.split(split_ratio=0.7, random_state=42,shuffle=True)
            len(train_cd), len(test_cd)
            # (700, 300)
            ```
        """
        train_paths, test_paths = train_test_split(
            data=self.image_paths,
            train_ratio=split_ratio,
            random_state=random_state,
            shuffle=shuffle,
        )

        train_input: Union[List[str], Dict[str, np.ndarray]]
        test_input: Union[List[str], Dict[str, np.ndarray]]
        if self._images_in_memory:
            train_input = {path: self._images_in_memory[path] for path in train_paths}
            test_input = {path: self._images_in_memory[path] for path in test_paths}
        else:
            train_input = train_paths
            test_input = test_paths
        train_annotations = {path: self.annotations[path] for path in train_paths}
        test_annotations = {path: self.annotations[path] for path in test_paths}

        train_dataset = ClassificationDataset(
            classes=self.classes,
            images=train_input,
            annotations=train_annotations,
        )
        test_dataset = ClassificationDataset(
            classes=self.classes,
            images=test_input,
            annotations=test_annotations,
        )

        return train_dataset, test_dataset

    def as_folder_structure(self, root_directory_path: str) -> None:
        """
        Saves the dataset as a multi-class folder structure.

        Args:
            root_directory_path (str): The path to the directory
                where the dataset will be saved.
        """
        os.makedirs(root_directory_path, exist_ok=True)

        for class_name in self.classes:
            os.makedirs(os.path.join(root_directory_path, class_name), exist_ok=True)

        for image_save_path, image, annotation in self:
            image_name = Path(image_save_path).name
            class_id = (
                annotation.class_id[0]
                if annotation.confidence is None
                else annotation.get_top_k(1)[0][0]
            )
            class_name = self.classes[class_id]
            image_save_path = os.path.join(root_directory_path, class_name, image_name)
            cv2.imwrite(image_save_path, image)

    @classmethod
    def from_folder_structure(cls, root_directory_path: str) -> ClassificationDataset:
        """
        Load data from a multiclass folder structure into a ClassificationDataset.

        Args:
            root_directory_path (str): The path to the dataset directory.

        Returns:
            ClassificationDataset: The dataset.

        Examples:
            ```python
            import roboflow
            from roboflow import Roboflow
            import supervision as sv

            roboflow.login()
            rf = Roboflow()

            project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
            dataset = project.version(PROJECT_VERSION).download("folder")

            cd = sv.ClassificationDataset.from_folder_structure(
                root_directory_path=f"{dataset.location}/train"
            )
            ```
        """
        classes = os.listdir(root_directory_path)
        classes = sorted(set(classes))

        image_paths = []
        annotations = {}

        for class_name in classes:
            class_id = classes.index(class_name)

            for image in os.listdir(os.path.join(root_directory_path, class_name)):
                image_path = str(os.path.join(root_directory_path, class_name, image))
                image_paths.append(image_path)
                annotations[image_path] = Classifications(
                    class_id=np.array([class_id]),
                )

        return cls(
            classes=classes,
            images=image_paths,
            annotations=annotations,
        )
