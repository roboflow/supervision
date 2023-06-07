import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from supervision.dataset.core import BaseDataset
from supervision.dataset.ultils import train_test_split


def _validate_class_ids(class_id: Any, n: int) -> None:
    """
    Ensure that class_id is a 1d np.ndarray with (n, ) shape.
    """
    is_valid = isinstance(class_id, np.ndarray) and class_id.shape == (n,)
    if not is_valid:
        raise ValueError("class_id must be 1d np.ndarray with (n, ) shape")


def _validate_confidence(confidence: Any, n: int) -> None:
    """
    Ensure that confidence is a 1d np.ndarray with (n, ) shape.
    """
    is_valid = isinstance(confidence, np.ndarray) and confidence.shape == (n,)
    if not is_valid:
        raise ValueError("confidence must be 1d np.ndarray with (n, ) shape")


@dataclass
class Classifications:
    class_id: np.ndarray
    confidence: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """
        Validate the classification inputs.
        """
        n = len(self.class_id)

        _validate_class_ids(self.class_id, n)

        if self.confidence is not None:
            _validate_confidence(self.confidence, n)

    def get_top_k(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the top k class IDs and confidences, ordered in descending order by confidence.

        Args:
            k (int): The number of top class IDs and confidences to retrieve.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the top k class IDs and confidences.

        Example:
            ```python
            >>> import supervision as sv

            >>> cd = sv.ClassificationDataset(...)

            >>> cd.annotations["image.png"].get_top_k(1)

            (array([1]), array([0.9]))
        """
        if self.confidence is None:
            raise ValueError("confidence is None")

        confidence = self.confidence.copy()
        class_ids = self.class_id.copy()

        order = np.argsort(confidence.copy())[::-1]
        top_k_order = order[:k]
        top_k_class_id = class_ids[top_k_order]
        top_k_confidence = confidence[top_k_order]

        return top_k_class_id, top_k_confidence


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
    ) -> Tuple["ClassificationDataset", "ClassificationDataset"]:
        """
        Splits the dataset into two parts (training and testing) using the provided split_ratio.

        Args:
            split_ratio (float, optional): The ratio of the training set to the entire dataset. Default is 0.8.
            random_state (int, optional): The seed for the random number generator. This is used for reproducibility. Default is None.
            shuffle (bool, optional): Whether to shuffle the data before splitting. Default is True.

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

    def as_multiclass_folder_structure(
        self, root_directory_path: str, output_directory_path: str
    ) -> None:
        """
        Saves the dataset as a multi-class folder structure.

        Args:
            root_directory_path (str): The path to the root directory of the images with which you are working.
            output_directory_path (str): The path to the directory where the dataset will be saved.

        Example:
            ```python
            >>> import supervision as sv

            >>> cd = sv.ClassificationDataset(...)

            >>> cd.as_multiclass_folder_structure(
            ...     root_directory_path="./images",
            ...     output_directory_path="./out",
            ... )
        """
        if not os.path.exists(output_directory_path):
            os.makedirs(output_directory_path)

        for split in ["train", "test", "val"]:
            if not os.path.exists(output_directory_path + "/" + split):
                os.makedirs(output_directory_path + "/" + split)

        train, test = self.split(split_ratio=0.8)
        train, val = train.split(split_ratio=0.8)

        for split, dataset in zip(["train", "test", "val"], [train, test, val]):
            for class_name in self.classes:
                if not os.path.exists(
                    output_directory_path + "/" + split + "/" + class_name
                ):
                    os.makedirs(output_directory_path + "/" + split + "/" + class_name)

                for image in dataset.images:
                    if dataset.annotations[image].class_id[0] == self.classes.index(
                        class_name
                    ):
                        full_dir = os.path.join(root_directory_path, image)

                        cv2.imwrite(
                            output_directory_path
                            + "/"
                            + split
                            + "/"
                            + class_name
                            + "/"
                            + image,
                            cv2.imread(full_dir),
                        )

    @classmethod
    def from_multiclass_folder_structure(
        cls, root_directory_path: str
    ) -> "ClassificationDataset":
        """
        Load data from a multiclass folder structure into a ClassificationDataset.

        Args:
            root_directory_path (str): The path to the dataset directory.

        Returns:
            ClassificationDataset: The dataset.

        Example:
            ```python
            >>> import supervision as sv

            >>> cd = sv.ClassificationDataset.from_multiclass_folder_structure(
            ...     root_directory_path="./dataset",
            ... )
        """
        classes = os.listdir(root_directory_path + "/train")
        classes.sort()

        images = {}
        annotations = {}

        for split in ["train", "test", "val"]:
            for class_name in classes:
                for image in os.listdir(
                    root_directory_path + "/" + split + "/" + class_name
                ):
                    images[image] = np.array([[[0, 0, 0]]])
                    annotations[image] = Classifications(
                        class_id=np.array([classes.index(class_name)]),
                    )

        return cls(
            classes=classes,
            images=images,
            annotations=annotations,
        )


dataset = ClassificationDataset(
    classes=["cat", "dog", "bird"],
    images={
        "cat1.jpg": np.array([[[0, 0, 0]]]),
    },
    annotations={
        "cat1.jpg": Classifications(
            class_id=np.array([0, 1]),
            confidence=np.array([0.3, 0.9]),
        ),
    },
)
