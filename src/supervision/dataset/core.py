from __future__ import annotations

import contextlib
import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
from deprecate import deprecated, void
from tqdm.auto import tqdm

from supervision import _cv2 as cv2
from supervision.classification.core import Classifications
from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.dataset.formats.coco import (
    _load_coco_annotations,
    save_coco_annotations,
)
from supervision.dataset.formats.coco import (
    load_coco_annotations as _load_coco_annotations_public,
)
from supervision.dataset.formats.createml import (
    load_createml_annotations,
    save_createml_annotations,
)
from supervision.dataset.formats.labelme import (
    load_labelme_annotations,
    save_labelme_annotations,
)
from supervision.dataset.formats.pascal_voc import (
    load_pascal_voc_annotations,
    save_pascal_voc_annotations,
)
from supervision.dataset.formats.yolo import (
    _extract_class_ids_from_data,
    _extract_class_names_from_data,
    _read_data_yaml,
    load_yolo_annotations,
    save_data_yaml,
    save_yolo_annotations,
)
from supervision.dataset.formats.yolo_pose import (
    _check_yolo_pose_export,
    load_yolo_pose_annotations,
    save_pose_data_yaml,
    save_yolo_pose_annotations,
)
from supervision.dataset.utils import (
    _validate_flip_idx,
    build_class_index_mapping,
    check_no_basename_collisions,
    map_detections_class_id,
    map_key_points_class_id,
    merge_class_lists,
    save_dataset_images,
    train_test_split,
)
from supervision.detection.core import Detections
from supervision.key_points.core import KeyPoints
from supervision.utils.file import save_yaml_file
from supervision.utils.internal import _DEPRECATION_WARNING_STREAM, warn_deprecated
from supervision.utils.iterables import find_duplicates

_IMAGE_FILE_EXTENSIONS = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)

#: Warning text for the copy of the COCO loader that this module exposed as a side
#: effect of using it. The loader itself stays public in its own module.
_COCO_LOADER_MODULE_DEPRECATION = (
    "`%(source_name)s` imported from `supervision.dataset.core` is deprecated in "
    "`%(deprecated_in)s` and will be removed in `%(remove_in)s`. Import it from "
    "`supervision.dataset.formats.coco` instead."
)


@deprecated(  # type: ignore[untyped-decorator]
    target=_load_coco_annotations_public,
    deprecated_in="0.32.0",
    remove_in="0.35.0",
    stream=_DEPRECATION_WARNING_STREAM,
    template_mgs=_COCO_LOADER_MODULE_DEPRECATION,
)
def load_coco_annotations(
    images_directory_path: str,
    annotations_path: str,
    force_masks: bool = False,
    use_iscrowd: bool = True,
    show_progress: bool = False,
    *,
    mask_rasterizer: Literal["supervision", "pycocotools"] | None = None,
) -> tuple[list[str], list[str], dict[str, Detections]]:
    """Deprecated: import `load_coco_annotations` from its own module.

    Releases before `0.32.0` exposed this loader from this module, so the name stays
    importable here until `0.35.0`.
    """
    return void(  # type: ignore[no-any-return]
        images_directory_path,
        annotations_path,
        force_masks,
        use_iscrowd,
        show_progress,
        mask_rasterizer=mask_rasterizer,
    )


def _validate_and_name_class_ids(
    annotations: dict[str, Detections] | dict[str, KeyPoints],
    classes: list[str],
    kind: str,
) -> None:
    """Check every annotation's class ids against the class list and name them.

    Shared by :class:`~supervision.dataset.core.DetectionDataset` and
    :class:`~supervision.dataset.core.KeyPointsDataset`, so both enforce the same
    class-id invariant. Annotations without ``class_id`` are skipped. Each
    annotation with class ids gains ``data['class_name']`` in place.

    Args:
        annotations: Annotations keyed by image path; mutated in place.
        classes: Dataset class names, ordered by class index.
        kind: Annotation noun for error messages, such as ``"Detection"``.

    Raises:
        ValueError: If an annotation carries non-integer class ids, or a class id
            outside the range of ``classes``; the message names the image.
    """
    np_classes = np.array(classes)
    for image_path, annotation in annotations.items():
        class_ids = annotation.class_id
        if class_ids is None:
            continue
        if not np.issubdtype(class_ids.dtype, np.integer):
            raise ValueError(
                f"{kind} annotation for image {image_path!r} contains "
                f"non-integer class_id values with dtype {class_ids.dtype}."
            )

        invalid_class_ids = class_ids[(class_ids < 0) | (class_ids >= len(classes))]
        if len(invalid_class_ids) > 0:
            valid_range = "empty" if len(classes) == 0 else f"[0, {len(classes) - 1}]"
            raise ValueError(
                f"{kind} annotation for image {image_path!r} contains "
                f"class_id {int(invalid_class_ids[0])}, outside the valid "
                f"range {valid_range} for {len(classes)} classes."
            )

        annotation.data[CLASS_NAME_DATA_FIELD] = np_classes[class_ids]


def _check_key_points_carry_class_id(annotations: dict[str, KeyPoints]) -> None:
    """Reject non-empty key points that carry no ``class_id``.

    A YOLO pose label row starts with the class id, so a skeleton without one
    cannot be exported. Checking up front keeps
    :meth:`~supervision.dataset.core.KeyPointsDataset.as_yolo` from failing
    halfway, after images were already written. Empty key points, such as
    :meth:`~supervision.key_points.core.KeyPoints.empty`, write no row and pass.

    Args:
        annotations: Key points keyed by image path.

    Raises:
        ValueError: If a non-empty annotation has ``class_id=None``; the message
            names the image.
    """
    for image_path, annotation in annotations.items():
        if annotation.class_id is None and len(annotation) > 0:
            raise ValueError(
                f"Key point annotation for image {image_path!r} holds "
                f"{len(annotation)} skeletons but no class_id; every skeleton of a "
                "KeyPointsDataset needs a class id to be written as a YOLO pose label."
            )


class BaseDataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def split(
        self,
        split_ratio: float = 0.8,
        random_state: int | None = None,
        shuffle: bool = True,
    ) -> tuple[BaseDataset, BaseDataset]:
        pass


def _yolo_source_mapping(
    data_before: dict[str, Any],
    data_after: dict[str, Any],
    classes: list[str],
    path: str,
) -> tuple[tuple[str, ...], tuple[int, ...]] | None:
    """Retain a nonstandard source mapping only when the YAML stayed consistent."""
    if data_before.get("names") != data_after.get("names"):
        raise ValueError(f"YOLO class names changed while reading '{path}'.")
    names = _extract_class_names_from_data(data_after, path)
    if names != classes:
        raise ValueError(f"YOLO class names changed while reading '{path}'.")
    class_ids = _extract_class_ids_from_data(data_after)
    if class_ids == list(range(len(classes))):
        return None
    return tuple(classes), tuple(class_ids)


def _prepare_yolo_export(
    dataset: DetectionDataset | KeyPointsDataset,
) -> DetectionDataset | KeyPointsDataset:
    """Prepare annotation-only export against the original YOLO class mapping."""
    if dataset._yolo_mapping_ambiguous:
        raise ValueError("Merged YOLO mappings differ; export a new data.yaml.")
    mapping = dataset._yolo_source_mapping
    if mapping is None:
        return dataset
    names, source_ids = mapping
    if tuple(dataset.classes) != names:
        raise ValueError("YOLO classes changed; export a new data.yaml.")
    for source_path in dataset._yolo_source_paths:
        if not Path(source_path).exists():
            continue
        source_data = _read_data_yaml(source_path)
        current_names = _extract_class_names_from_data(source_data, source_path)
        current_ids = _extract_class_ids_from_data(source_data)
        # Compare id-to-name pairs, since a merged mapping follows merged class order.
        if dict(zip(current_ids, current_names)) != dict(zip(source_ids, names)):
            raise ValueError(
                "Source YOLO class mapping changed; export a new data.yaml."
            )
    for image_path, annotation in dataset.annotations.items():
        ids = annotation.class_id
        if ids is None:
            continue
        if not np.issubdtype(ids.dtype, np.integer) or np.any(
            (ids < 0) | (ids >= len(source_ids))
        ):
            raise ValueError(f"Invalid class ids for YOLO export in '{image_path}'.")
    export_dataset = copy(dataset)
    export_dataset._yolo_export_class_ids = list(source_ids)
    return export_dataset


def _shared_yolo_mapping(
    classes: list[str],
    datasets: Sequence[DetectionDataset] | Sequence[KeyPointsDataset],
) -> tuple[tuple[str, ...], tuple[int, ...]] | None:
    """Re-key the source mapping every dataset shares to ``classes`` order.

    Returns ``None`` when any dataset lacks a source mapping, when the id-to-name pairs
    differ between datasets, or when ``classes`` names a class without a source id.
    """
    id_to_names: list[dict[int, str]] = []
    for dataset in datasets:
        mapping = dataset._yolo_source_mapping
        # Classes edited after loading no longer show which source id a label had.
        if mapping is None or tuple(dataset.classes) != mapping[0]:
            return None
        names, source_ids = mapping
        id_to_names.append(dict(zip(source_ids, names)))
    if any(id_to_name != id_to_names[0] for id_to_name in id_to_names):
        return None
    name_to_id = {name: class_id for class_id, name in id_to_names[0].items()}
    if len(name_to_id) != len(id_to_names[0]):
        return None
    if set(classes) != set(name_to_id):
        return None
    return tuple(classes), tuple(name_to_id[name] for name in classes)


def _inherit_yolo_mapping(
    merged: DetectionDataset | KeyPointsDataset,
    datasets: Sequence[DetectionDataset] | Sequence[KeyPointsDataset],
) -> None:
    """Preserve compatible source mappings or require a fresh YAML after merging.

    Compatibility compares id-to-name pairs, never the ``data.yaml`` path or the class-
    list order, so equal mappings stay reusable after merge sorts the classes.
    """
    if any(dataset._yolo_mapping_ambiguous for dataset in datasets):
        merged._yolo_mapping_ambiguous = True
        return
    if all(dataset._yolo_source_mapping is None for dataset in datasets):
        return
    mapping = _shared_yolo_mapping(merged.classes, datasets)
    if mapping is None:
        merged._yolo_mapping_ambiguous = True
        return
    merged._yolo_source_mapping = mapping
    merged._yolo_source_paths = tuple(
        dict.fromkeys(
            path for dataset in datasets for path in dataset._yolo_source_paths
        )
    )


def _save_source_yolo_yaml(
    dataset: DetectionDataset | KeyPointsDataset, data_yaml_path: str
) -> None:
    """Write source class IDs when labels are not being rewritten."""
    mapping = dataset._yolo_source_mapping
    if mapping is None:
        raise ValueError("Source YOLO class mapping is unavailable.")
    names, source_ids = mapping
    # A merged mapping follows merged class order; write the source id order.
    source_names = dict(sorted(zip(source_ids, names)))
    data: dict[str, Any] = {"nc": len(names), "names": source_names}
    if isinstance(dataset, KeyPointsDataset):
        data["kpt_shape"] = [dataset.keypoint_count, dataset.value_dim]
        if dataset.flip_idx is not None:
            data["flip_idx"] = dataset.flip_idx
    Path(data_yaml_path).parent.mkdir(parents=True, exist_ok=True)
    save_yaml_file(data=data, file_path=data_yaml_path)


class DetectionDataset(BaseDataset):
    """Contains information about a detection dataset.

    Handles lazy image loading and annotation retrieval, dataset splitting, conversions
    into multiple formats.

    Attributes:
        classes: List containing dataset class names.
        images:
            Accepts a list of image paths. Passing a dict
            (``Dict[str, np.ndarray]``) is deprecated in ``0.30.0`` and will
            be removed in ``0.33.0``; use a list of paths instead.
            When a list of paths is provided, images are loaded lazily on
            demand, which is more memory-efficient.
        annotations: Dictionary mapping
            image path to annotations. The dictionary keys match
            match the keys in `images` or entries in the list of
            image paths.
    """

    def __init__(
        self,
        classes: list[str],
        images: list[str] | dict[str, npt.NDArray[np.uint8]],
        annotations: dict[str, Detections],
    ) -> None:
        self.classes = classes
        self._yolo_source_mapping: tuple[tuple[str, ...], tuple[int, ...]] | None = None
        self._yolo_source_paths: tuple[str, ...] = ()
        self._yolo_mapping_ambiguous = False
        self._yolo_export_class_ids: list[int] | None = None
        self._yolo_normalized_export = False

        if set(images) != set(annotations):
            raise ValueError(
                "The keys of the images and annotations dictionaries must match."
            )
        self.annotations = {
            image_path: deepcopy(annotation)
            for image_path, annotation in annotations.items()
        }

        _validate_and_name_class_ids(
            annotations=self.annotations, classes=self.classes, kind="Detection"
        )

        # Eliminate duplicates while preserving order
        self.image_paths = list(dict.fromkeys(images))

        self._images_in_memory: dict[str, npt.NDArray[np.uint8]] = {}
        if isinstance(images, dict):
            self._images_in_memory = images
            warn_deprecated(
                "Passing a `Dict[str, np.ndarray]` into `DetectionDataset` is "
                "deprecated in `0.30.0` and will be removed in `0.33.0`. Use "
                "a list of paths `List[str]` instead."
            )

    def _get_image(self, image_path: str) -> npt.NDArray[np.uint8]:
        """Assumes that image is in dataset."""
        if self._images_in_memory:
            return self._images_in_memory[image_path]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from path: {image_path}")
        return cast(npt.NDArray[np.uint8], image)

    def __len__(self) -> int:
        return len(self._images_in_memory) or len(self.image_paths)

    def __getitem__(self, i: int) -> tuple[str, npt.NDArray[np.uint8], Detections]:
        """
        Returns:
            The image path, image data,
                and its corresponding annotation at index i.
        """
        image_path = self.image_paths[i]
        image = self._get_image(image_path)
        annotation = self.annotations[image_path]
        return image_path, image, annotation

    def __iter__(self) -> Iterator[tuple[str, npt.NDArray[np.uint8], Detections]]:
        """Iterate over the images and annotations in the dataset.

        Yields:
            Tuples containing the image path, image data, and its annotation.
        """
        for i in range(len(self)):
            image_path, image, annotation = self[i]
            yield image_path, image, annotation

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DetectionDataset):
            return False

        if self.classes != other.classes:
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
        random_state: int | None = None,
        shuffle: bool = True,
    ) -> tuple[DetectionDataset, DetectionDataset]:
        """Splits the dataset into two parts (training and testing) using the provided
        split_ratio. The input dataset is not mutated.

        Args:
            split_ratio: The ratio of the training set to the entire dataset,
                within the inclusive range `[0, 1]`. `0` sends every image to
                the testing dataset and `1` sends every image to the training
                dataset.
            random_state: The seed for the random number generator.
                This is used for reproducibility.
            shuffle: Whether to shuffle the data before splitting.

        Returns:
            A tuple containing
                the training and testing datasets.

        Raises:
            ValueError: If `split_ratio` is outside `[0, 1]` or is not a finite
                number.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> ds = sv.DetectionDataset(
            ...     classes=['dog', 'person'],
            ...     images={
            ...         'img1.jpg': np.zeros((100, 100, 3), dtype=np.uint8),
            ...         'img2.jpg': np.zeros((100, 100, 3), dtype=np.uint8),
            ...     },
            ...     annotations={
            ...         'img1.jpg': sv.Detections(xyxy=np.array([[10, 10, 20, 20]])),
            ...         'img2.jpg': sv.Detections(xyxy=np.array([[30, 30, 40, 40]])),
            ...     }
            ... )
            >>> train_ds, test_ds = ds.split(split_ratio=0.5, random_state=42)
            >>> len(train_ds), len(test_ds)
            (1, 1)

            ```
        """

        train_paths, test_paths = train_test_split(
            data=self.image_paths,
            train_ratio=split_ratio,
            random_state=random_state,
            shuffle=shuffle,
        )

        train_annotations = {path: self.annotations[path] for path in train_paths}
        test_annotations = {path: self.annotations[path] for path in test_paths}

        train_dataset = DetectionDataset(
            classes=self.classes,
            images=train_paths,
            annotations=train_annotations,
        )
        test_dataset = DetectionDataset(
            classes=self.classes,
            images=test_paths,
            annotations=test_annotations,
        )
        for subset in (train_dataset, test_dataset):
            subset._yolo_source_mapping = self._yolo_source_mapping
            subset._yolo_source_paths = self._yolo_source_paths
            subset._yolo_mapping_ambiguous = self._yolo_mapping_ambiguous
        if self._images_in_memory:
            train_dataset._images_in_memory = {
                path: self._images_in_memory[path] for path in train_paths
            }
            test_dataset._images_in_memory = {
                path: self._images_in_memory[path] for path in test_paths
            }
        return train_dataset, test_dataset

    @classmethod
    def merge(cls, dataset_list: list[DetectionDataset]) -> DetectionDataset:
        """Merge a list of :class:`~supervision.dataset.core.DetectionDataset` objects
        into a single :class:`~supervision.dataset.core.DetectionDataset` object.

        This method takes a list of
        :class:`~supervision.dataset.core.DetectionDataset` objects and combines
        their respective fields (`classes`, `images`, `annotations`) into a
        single :class:`~supervision.dataset.core.DetectionDataset` object.

        Args:
            dataset_list: A list of :class:`~supervision.dataset.core.DetectionDataset`
                objects to merge.

        Returns:
            A single :class:`~supervision.dataset.core.DetectionDataset` object
            containing the merged data from the input list.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> ds_1 = sv.DetectionDataset(
            ...     classes=['dog', 'person'],
            ...     images={'img1.jpg': np.zeros((100, 100, 3), dtype=np.uint8)},
            ...     annotations={'img1.jpg': sv.Detections.empty()}
            ... )
            >>> len(ds_1)
            1
            >>> ds_1.classes
            ['dog', 'person']
            >>> ds_2 = sv.DetectionDataset(
            ...     classes=['cat'],
            ...     images={'img2.jpg': np.zeros((100, 100, 3), dtype=np.uint8)},
            ...     annotations={'img2.jpg': sv.Detections.empty()}
            ... )
            >>> len(ds_2)
            1
            >>> ds_2.classes
            ['cat']
            >>> ds_merged = sv.DetectionDataset.merge([ds_1, ds_2])
            >>> len(ds_merged)
            2
            >>> ds_merged.classes
            ['cat', 'dog', 'person']

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
                if annotations[image_path].class_id is None:
                    continue
                annotations[image_path] = map_detections_class_id(
                    source_to_target_mapping=class_index_mapping,
                    detections=annotations[image_path],
                )

        merged_dataset = cls(
            classes=classes,
            images=image_paths,
            annotations=annotations,
        )
        if all_in_memory:
            merged_dataset._images_in_memory = images_in_memory
        _inherit_yolo_mapping(merged_dataset, dataset_list)
        return merged_dataset

    def as_pascal_voc(
        self,
        images_directory_path: str | None = None,
        annotations_directory_path: str | None = None,
        min_image_area_percentage: float = 0.0,
        max_image_area_percentage: float = 1.0,
        approximation_percentage: float = 0.0,
        show_progress: bool = False,
    ) -> None:
        """Exports the dataset to PASCAL VOC format.

        This method saves the images and their corresponding annotations in PASCAL VOC
        format. Both output layouts are preflighted before any files are written so a
        collision in either target fails without partial output.

        Args:
            images_directory_path: The path to the directory
                where the images should be saved.
                If not provided, images will not be saved.
            annotations_directory_path: The path to
                the directory where the annotations in PASCAL VOC format should be
                saved. If not provided, annotations will not be saved.
            min_image_area_percentage: The minimum percentage of
                detection area relative to
                the image area for a detection to be included.
                Argument is used only for segmentation datasets.
            max_image_area_percentage: The maximum percentage
                of detection area relative to
                the image area for a detection to be included.
                Argument is used only for segmentation datasets.
            approximation_percentage: The percentage of
                polygon points to be removed from the input polygon,
                in the range [0, 1). Argument is used only for segmentation datasets.
            show_progress: If True, display a progress bar during saving.

        Raises:
            ValueError: If two image paths share the same basename (when
                images_directory_path is set) or the same stem (when
                annotations_directory_path is set), which would cause one
                output file to overwrite another. Rename images to ensure
                unique basenames before exporting a merged dataset.
        """
        if images_directory_path:
            check_no_basename_collisions(
                image_paths=self.image_paths,
                key=lambda image_path: Path(image_path).name,
                output_kind="image",
            )
        if annotations_directory_path:
            check_no_basename_collisions(
                image_paths=self.image_paths,
                key=lambda image_path: f"{Path(image_path).stem}.xml",
                output_kind="Pascal VOC annotation",
            )

        if images_directory_path:
            save_dataset_images(
                dataset=self,
                images_directory_path=images_directory_path,
                show_progress=show_progress,
            )
        if annotations_directory_path:
            save_pascal_voc_annotations(
                dataset=self,
                annotations_directory_path=annotations_directory_path,
                min_image_area_percentage=min_image_area_percentage,
                max_image_area_percentage=max_image_area_percentage,
                approximation_percentage=approximation_percentage,
                show_progress=show_progress,
            )

    @classmethod
    def from_pascal_voc(
        cls,
        images_directory_path: str,
        annotations_directory_path: str,
        force_masks: bool = False,
        show_progress: bool = False,
    ) -> DetectionDataset:
        """Creates a Dataset instance from PASCAL VOC formatted data.

        Args:
            images_directory_path: Path to the directory containing the images.
            annotations_directory_path: Path to the directory
                containing the PASCAL VOC XML annotations.
            force_masks: If True, forces masks to
                be loaded for all annotations, regardless of whether they are present.
            show_progress: If True, display a progress bar during loading.

        Returns:
            A DetectionDataset instance containing
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
                annotations_directory_path=f"{dataset.location}/train/labels",
                # pass show_progress=True to enable a tqdm progress bar
            )

            ds.classes
            # ['dog', 'person']
            ```
        """

        classes, image_paths, annotations = load_pascal_voc_annotations(
            images_directory_path=images_directory_path,
            annotations_directory_path=annotations_directory_path,
            force_masks=force_masks,
            show_progress=show_progress,
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
        show_progress: bool = False,
        *,
        compact_masks: bool = False,
    ) -> DetectionDataset:
        """Creates a Dataset instance from YOLO formatted data.

        When `data.yaml` declares `kpt_shape: [K, D]`, as Ultralytics pose
        datasets do, a row of exactly `class x y w h` followed by `K * D`
        keypoint values is read as its box; the keypoints are not loaded, and
        rows of other lengths are read as before. A polygon row of that same
        length, with or without a trailing confidence, is read as a box too,
        as Ultralytics does. Pose rows with a trailing confidence or tracker id,
        or with a keypoint count that does not match `kpt_shape`, are not
        supported and are parsed as before. A `kpt_shape` other than `[K, 2]`
        or `[K, 3]` raises `ValueError`.

        Loaded `class_id` values index `classes`. When `names` is a dict whose
        keys start above `0` or skip values, such as `{1: cat, 2: dog}`, each
        label id is mapped to the name declared for it, so label id `1` loads
        as class `0`.

        Args:
            images_directory_path: The path to the
                directory containing the images.
            annotations_directory_path: The path to the directory
                containing the YOLO annotation files.
            data_yaml_path: The path to the data
                YAML file containing class information and, for pose
                datasets, `kpt_shape`.
            force_masks: If True, forces
                masks to be loaded for all annotations,
                regardless of whether they are present.
            is_obb: If True, loads the annotations in OBB format.
                OBB annotations are defined as `[class_id, x, y, x, y, x, y, x, y]`,
                where pairs of [x, y] are box corners.
            show_progress: If True, display a progress bar during loading.
            compact_masks: If True, store segmentation masks as
                :class:`~supervision.detection.compact_mask.CompactMask` instead
                of dense NumPy arrays. Each mask is rasterized full-frame and
                immediately encoded within its rounded polygon bounds without
                changing pixels. This usually reduces retained memory, but
                compression depends on mask complexity and can increase memory
                for highly fragmented masks. Does not enable masks for box-only
                datasets; combine with `force_masks=True` to generate compact box
                masks. OBB annotations never generate masks.

        Returns:
            A DetectionDataset instance
                containing the loaded images and annotations.

        Raises:
            ValueError: If `data.yaml` is not a mapping; if its `names` is
                missing, not a list or dict, a dict mixing numeric and
                non-numeric keys, declared twice, or a dict repeating a class
                id; if a label uses a class id that `names` does not declare;
                or if its `kpt_shape` is present but not
                `[K, 2]` or `[K, 3]`, `kpt_shape: null` included. Also if an
                annotation file contains an invalid line, such as one with too
                few tokens, a class id that is not a whole number, a
                non-numeric coordinate, or a box with a negative width or
                height; the message names the offending file.

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
                data_yaml_path=f"{dataset.location}/data.yaml",
                # pass show_progress=True to enable a tqdm progress bar
                # pass compact_masks=True to reduce segmentation mask storage
            )

            ds.classes
            # ['dog', 'person']
            ```
        """
        source_data = _read_data_yaml(file_path=data_yaml_path)
        classes, image_paths, annotations = load_yolo_annotations(
            images_directory_path=images_directory_path,
            annotations_directory_path=annotations_directory_path,
            data_yaml_path=data_yaml_path,
            force_masks=force_masks,
            is_obb=is_obb,
            show_progress=show_progress,
            compact_masks=compact_masks,
        )
        dataset = DetectionDataset(
            classes=classes, images=image_paths, annotations=annotations
        )
        dataset._yolo_source_mapping = _yolo_source_mapping(
            source_data,
            _read_data_yaml(file_path=data_yaml_path),
            classes,
            data_yaml_path,
        )
        dataset._yolo_source_paths = (str(Path(data_yaml_path).resolve()),)
        return dataset

    def as_yolo(
        self,
        images_directory_path: str | None = None,
        annotations_directory_path: str | None = None,
        data_yaml_path: str | None = None,
        min_image_area_percentage: float = 0.0,
        max_image_area_percentage: float = 1.0,
        approximation_percentage: float = 0.0,
        is_obb: bool = False,
        show_progress: bool = False,
        ignore_crowd: bool = True,
        merge_segments: bool = True,
    ) -> None:
        """Exports the dataset to YOLO format.

        This method saves the images and their corresponding annotations in YOLO format.

        Args:
            images_directory_path: The path to the
                directory where the images should be saved.
                If not provided, images will not be saved.
            annotations_directory_path: The path to the
                directory where the annotations in
                YOLO format should be saved. If not provided,
                annotations will not be saved.
            data_yaml_path: The path where the data.yaml
                file should be saved.
                If not provided, the file will not be saved. For a dataset loaded
                from YOLO with non-contiguous class IDs, annotation-only export
                keeps those IDs for use with the source YAML; YAML-only export
                keeps the source IDs for existing labels. When both are written,
                IDs are normalized. If class names have changed or merged source
                mappings disagree, export both annotations and a new YAML.
            min_image_area_percentage: The minimum percentage of
                detection area relative to
                the image area for a detection to be included.
                Argument is used only for segmentation datasets.
            max_image_area_percentage: The maximum percentage
                of detection area relative to
                the image area for a detection to be included.
                Argument is used only for segmentation datasets.
            approximation_percentage: The percentage of polygon points to
                be removed from the input polygon, in the range [0, 1).
                This is useful for simplifying the annotations.
                Argument is used only for segmentation datasets.
            is_obb: If True, exports annotations in OBB format
                (`class_id x1 y1 x2 y2 x3 y3 x4 y4`) using the oriented
                corners stored in `detections.data["xyxyxyxy"]`. Mirrors
                `from_yolo(..., is_obb=True)`. Masks are ignored when
                `is_obb=True`.
            show_progress: If True, display a progress bar during saving.
            ignore_crowd: When True (default), detections marked with a non-zero
                `iscrowd` flag in `detections.data` are omitted from the export.
                When False, those detections are included.
            merge_segments: When True (default), multiple disconnected polygon
                contours belonging to a single detection mask are joined using
                zero-width seams, emitting exactly one YOLO polygon line per
                detection. When False, each disconnected contour is exported on a
                separate line.

        Raises:
            ValueError: If two image paths share the same basename (when
                images_directory_path is set) or the same annotation
                file name (when annotations_directory_path is set),
                which would cause one output file to overwrite another, or if
                source class mapping cannot be safely reused without a new YAML.
        """
        export_dataset = self
        if (data_yaml_path is None) != (annotations_directory_path is None):
            export_dataset = cast(DetectionDataset, _prepare_yolo_export(self))
        elif data_yaml_path is not None:
            export_dataset = copy(self)
            export_dataset._yolo_normalized_export = True
        if is_obb and (
            min_image_area_percentage != 0.0
            or max_image_area_percentage != 1.0
            or approximation_percentage != 0.0
        ):
            import warnings

            warnings.warn(
                "`min_image_area_percentage`, `max_image_area_percentage`, and "
                "`approximation_percentage` have no effect when `is_obb=True`; "
                "OBB annotations use corner coordinates directly.",
                UserWarning,
                stacklevel=2,
            )
        # Pre-flight: validate output uniqueness before writing any file
        if images_directory_path:
            check_no_basename_collisions(
                image_paths=self.image_paths,
                key=lambda image_path: Path(image_path).name,
                output_kind="image",
            )
        if annotations_directory_path:
            check_no_basename_collisions(
                image_paths=self.image_paths,
                key=lambda image_path: Path(image_path).stem + ".txt",
                output_kind="YOLO annotation",
            )

        if images_directory_path is not None:
            save_dataset_images(
                dataset=self,
                images_directory_path=images_directory_path,
                show_progress=show_progress,
            )
        if annotations_directory_path is not None:
            save_yolo_annotations(
                dataset=export_dataset,
                annotations_directory_path=annotations_directory_path,
                min_image_area_percentage=min_image_area_percentage,
                max_image_area_percentage=max_image_area_percentage,
                approximation_percentage=approximation_percentage,
                show_progress=show_progress,
                is_obb=is_obb,
                ignore_crowd=ignore_crowd,
                merge_segments=merge_segments,
            )
        if data_yaml_path is not None:
            if annotations_directory_path is None and self._yolo_source_mapping:
                _save_source_yolo_yaml(self, data_yaml_path)
            else:
                save_data_yaml(data_yaml_path=data_yaml_path, classes=self.classes)

    @classmethod
    def from_labelme(
        cls,
        images_directory_path: str,
        annotations_directory_path: str,
        force_masks: bool = False,
    ) -> DetectionDataset:
        """Creates a Dataset instance from LabelMe formatted data.

        LabelMe stores one JSON file per image, each containing a list of
        ``shapes``. ``rectangle`` shapes are loaded as bounding boxes and
        ``polygon`` shapes as masks (with their bounding boxes); other shape
        types are skipped. Class names are inferred from the labels present in
        the files. When an image file contains a ``polygon`` shape, or when
        ``force_masks=True`` is set, both ``rectangle`` and ``polygon`` shapes
        produce masks: rectangles via a four-corner polygon fill.
        Shapes with the same label and non-null ``group_id`` are combined into
        one detection. Shapes without a group ID remain independent.

        Args:
            images_directory_path: The path to the
                directory containing the images.
            annotations_directory_path: The path to the directory
                containing the LabelMe ``.json`` annotation files.
            force_masks: If True, forces masks to be loaded for all
                annotations, regardless of whether polygon shapes are present.
                Requires ``imageWidth`` and ``imageHeight`` in every JSON file.

        Returns:
            A DetectionDataset instance containing
                the loaded images and annotations.

        Raises:
            ValueError: If an annotation is malformed - for example
                ``imagePath`` is empty or resolves to ``..``, a shape is
                missing its ``label`` or ``points``, or a mask is required but
                ``imageWidth`` / ``imageHeight`` are missing or zero.

        Examples:
            ```python
            import supervision as sv

            ds = sv.DetectionDataset.from_labelme(
                images_directory_path="<IMAGES_DIRECTORY_PATH>",
                annotations_directory_path="<ANNOTATIONS_DIRECTORY_PATH>",
            )

            ds.classes
            # ['dog', 'person']
            ```
        """
        classes, image_paths, annotations = load_labelme_annotations(
            images_directory_path=images_directory_path,
            annotations_directory_path=annotations_directory_path,
            force_masks=force_masks,
        )
        return DetectionDataset(
            classes=classes, images=image_paths, annotations=annotations
        )

    def as_labelme(
        self,
        images_directory_path: str | None = None,
        annotations_directory_path: str | None = None,
    ) -> None:
        """Exports the dataset to LabelMe format. This method saves the images and their
        corresponding annotations as per-image LabelMe ``.json`` files. Masked
        detections are written as ``polygon`` shapes whose vertices approximate the mask
        contour, so masks are not bit-exact on round-trip. Because the bounding box is
        recomputed from the quantized polygon contour on re-import, bounding boxes for
        masked detections may also shift by approximately one pixel after a save-load
        cycle. Disconnected components of a mask share a ``group_id`` so they load back
        as one instance.

        Args:
            images_directory_path: The path to the directory
                where the images should be saved.
                If not provided, images will not be saved.
            annotations_directory_path: The path to the directory where the
                LabelMe ``.json`` files should be saved.
                If not provided, annotations will not be saved.

        Examples:
            ```python
            import supervision as sv

            ds = sv.DetectionDataset(...)

            ds.as_labelme(
                images_directory_path="<IMAGES_DIRECTORY_PATH>",
                annotations_directory_path="<ANNOTATIONS_DIRECTORY_PATH>",
            )
            ```
        """
        if images_directory_path is not None:
            save_dataset_images(
                dataset=self, images_directory_path=images_directory_path
            )
        if annotations_directory_path is not None:
            save_labelme_annotations(
                dataset=self,
                annotations_directory_path=annotations_directory_path,
            )

    @classmethod
    def from_createml(
        cls,
        images_directory_path: str,
        annotations_path: str,
        show_progress: bool = False,
    ) -> DetectionDataset:
        """Creates a Dataset instance from CreateML formatted data.

        CreateML stores object-detection annotations in a single JSON file as a
        list of per-image entries, with each box expressed as a pixel-space
        centre point plus width and height. Class names are inferred from the
        labels present in the file.

        Args:
            images_directory_path: The path to the directory containing the
                images.
            annotations_path: The path to the CreateML json annotation file.
            show_progress: If True, display a progress bar during loading.

        Returns:
            A DetectionDataset instance containing the loaded images and
            annotations.

        Examples:
            ```python
            import roboflow
            from roboflow import Roboflow
            import supervision as sv

            roboflow.login()
            rf = Roboflow()

            project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
            dataset = project.version(PROJECT_VERSION).download("createml")

            ds = sv.DetectionDataset.from_createml(
                images_directory_path=f"{dataset.location}/train",
                annotations_path=f"{dataset.location}/train/_annotations.createml.json",
            )

            ds.classes
            # ['dog', 'person']
            ```
        """
        classes, image_paths, annotations = load_createml_annotations(
            images_directory_path=images_directory_path,
            annotations_path=annotations_path,
            show_progress=show_progress,
        )
        return DetectionDataset(
            classes=classes, images=image_paths, annotations=annotations
        )

    def as_createml(
        self,
        images_directory_path: str | None = None,
        annotations_path: str | None = None,
        show_progress: bool = False,
    ) -> None:
        """Exports the dataset to CreateML format. This method saves the images and
        their corresponding annotations in CreateML format.

        Args:
            images_directory_path: The path to the directory where the images
                should be saved. If not provided, images will not be saved.
            annotations_path: The path to the CreateML json annotation file.
                If not provided, the annotations will not be saved.
            show_progress: If True, display a progress bar while saving images.

        Returns:
            None. Side-effects only: writes images and/or annotation file.

        Examples:
            ```python
            import supervision as sv

            ds = sv.DetectionDataset(classes=["dog"], images=[], annotations={})
            ds.as_createml(
                images_directory_path="/tmp/images",
                annotations_path="/tmp/annotations.json",
            )
            ```
        """
        if images_directory_path is not None:
            save_dataset_images(
                dataset=self,
                images_directory_path=images_directory_path,
                show_progress=show_progress,
            )
        if annotations_path is not None:
            save_createml_annotations(
                dataset=self,
                annotations_path=annotations_path,
            )

    @classmethod
    def from_coco(
        cls,
        images_directory_path: str,
        annotations_path: str,
        force_masks: bool = False,
        show_progress: bool = False,
        *,
        use_iscrowd: bool = True,
        mask_rasterizer: Literal["supervision", "pycocotools"] | None = None,
    ) -> DetectionDataset:
        """Creates a Dataset instance from COCO formatted data.

        Args:
            images_directory_path: The path to the
                directory containing the images.
            annotations_path: The path to the json annotation files.
            force_masks: If True,
                forces masks to be loaded for all annotations,
                regardless of whether they are present.
            show_progress: If True, display a progress bar during loading.
            use_iscrowd: If True, includes COCO ``iscrowd`` and ``area``
                annotation fields in ``Detections.data``.
            mask_rasterizer: ``"supervision"`` uses rounded vertices with
                inclusive integer fill. ``"pycocotools"`` uses COCO's reference
                polygon rasterization with continuous vertices. Install the
                optional ``supervision[coco]`` extra for this mode; it is
                required even when the file has no polygon segmentation.
                ``None`` (default) behaves like ``"supervision"`` and warns
                when a loaded image has a non-empty polygon part. RLE masks are
                decoded identically in all modes.

        Returns:
            A DetectionDataset instance containing
                the loaded images and annotations.

        Raises:
            ValueError: If ``mask_rasterizer`` is not a supported value.
            ValueError: If a polygon vertex is not a finite number.
            ImportError: If ``mask_rasterizer="pycocotools"`` and the
                ``supervision[coco]`` extra is not installed. This is checked
                before any image is processed, even when the file has no
                polygon segmentation.

        Warns:
            SupervisionWarnings: When ``mask_rasterizer`` is omitted and a
                loaded image has a non-empty polygon part. Relying on the
                default is deprecated in ``0.32.0``.

        Note:
            COCO polygon vertices describe continuous boundaries, while the
            ``"supervision"`` rasterizer treats rounded vertices as inclusive
            pixel indices. ``"pycocotools"`` is the COCO convention: for
            well-formed polygons (nested lists of even-length parts with at
            least three vertices each) it matches the masks COCO evaluation
            builds. For example, a square from ``(1, 1)`` to ``(5, 5)``
            covers 16 reference pixels instead of 25 inclusive pixels. For
            other input the loader follows its own rules: a part with fewer
            than three vertices is skipped, a flat coordinate list is treated
            as one part, and an odd coordinate count raises ``ValueError``.
            The default changes from
            ``"supervision"`` to ``"pycocotools"`` no earlier than
            ``0.35.0``; pass ``mask_rasterizer`` explicitly to keep the
            current masks or to adopt the reference ones now. This option
            does not change polygon export: use RLE when exact mask
            preservation across COCO tools is required.

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
                # pass show_progress=True to enable a tqdm progress bar
            )

            ds.classes
            # ['dog', 'person']
            ```
        """
        # Call the shared implementation directly, as load_coco_annotations does,
        # so the default-rasterizer warning points at this method's caller.
        classes, images, annotations = _load_coco_annotations(
            images_directory_path=images_directory_path,
            annotations_path=annotations_path,
            force_masks=force_masks,
            use_iscrowd=use_iscrowd,
            show_progress=show_progress,
            mask_rasterizer=mask_rasterizer,
        )
        return DetectionDataset(classes=classes, images=images, annotations=annotations)

    def as_coco(
        self,
        images_directory_path: str | None = None,
        annotations_path: str | None = None,
        min_image_area_percentage: float = 0.0,
        max_image_area_percentage: float = 1.0,
        approximation_percentage: float = 0.0,
        starting_image_id: int = 1,
        starting_annotation_id: int = 1,
        show_progress: bool = False,
    ) -> tuple[int, int]:
        """Exports the dataset to COCO format. This method saves the images and their
        corresponding annotations in COCO format.

        !!! tip

            The format of the mask is determined automatically based on its structure:

            - If a mask contains multiple disconnected components or holes, it will be
            saved using the Run-Length Encoding (RLE) format for efficient storage and
            processing.
            - If a mask consists of a single, contiguous region without any holes, it
            will be encoded as a polygon, preserving the outline of the object.
            - If `detections.data` holds `iscrowd`, as it does after
            :meth:`~supervision.dataset.core.DetectionDataset.from_coco`, a mask with
            `iscrowd=0` is always encoded as polygons, with holes joined to their
            outline along zero-width seams where a seam is found.

            This automatic selection ensures that the masks are stored in the most
            appropriate and space-efficient format, complying with COCO dataset
            standards.

        Args:
            images_directory_path: The path to the directory
                where the images should be saved.
                If not provided, images will not be saved.
            annotations_path: The path to COCO annotation file.
            min_image_area_percentage: The minimum percentage of
                detection area relative to
                the image area for a detection to be included.
                Argument is used only for segmentation datasets.
            max_image_area_percentage: The maximum percentage of
                detection area relative to
                the image area for a detection to be included.
                Argument is used only for segmentation datasets.
            approximation_percentage: The percentage of polygon points
                to be removed from the input polygon,
                in the range [0, 1). This is useful for simplifying the annotations.
                Argument is used only for segmentation datasets.
            starting_image_id: First image id to assign in the exported file.
                Defaults to ``1``. Override when exporting multiple splits into
                a coordinated COCO collection so ids remain unique across the
                set (see example below).
            starting_annotation_id: First annotation id to assign in the
                exported file. Defaults to ``1``. Override for the same
                multi-split reason as ``starting_image_id``.
            show_progress: If True, display a progress bar during saving.

        Returns:
            A ``(next_image_id, next_annotation_id)`` tuple containing the
            first unused ids after this export. Feed them straight back into
            ``starting_image_id`` and ``starting_annotation_id`` on the next
            split so ids stay globally unique. When ``annotations_path`` is
            ``None`` (images-only export) the starting ids are returned
            unchanged so chaining still composes.

        Example:
            ```python
            # Exporting train, valid, and test splits with non-colliding ids
            # so the three annotation files can later be merged into one COCO.
            next_image_id, next_annotation_id = train_ds.as_coco(
                images_directory_path="out/train/images",
                annotations_path="out/train/annotations.json",
            )
            next_image_id, next_annotation_id = valid_ds.as_coco(
                images_directory_path="out/valid/images",
                annotations_path="out/valid/annotations.json",
                starting_image_id=next_image_id,
                starting_annotation_id=next_annotation_id,
            )
            _, _ = test_ds.as_coco(
                images_directory_path="out/test/images",
                annotations_path="out/test/annotations.json",
                starting_image_id=next_image_id,
                starting_annotation_id=next_annotation_id,
            )  # return value not needed — no further split
            ```
        """
        if images_directory_path is not None:
            save_dataset_images(
                dataset=self,
                images_directory_path=images_directory_path,
                show_progress=show_progress,
            )
        if annotations_path is not None:
            return save_coco_annotations(
                dataset=self,
                annotation_path=annotations_path,
                min_image_area_percentage=min_image_area_percentage,
                max_image_area_percentage=max_image_area_percentage,
                approximation_percentage=approximation_percentage,
                starting_image_id=starting_image_id,
                starting_annotation_id=starting_annotation_id,
                show_progress=show_progress,
            )
        return starting_image_id, starting_annotation_id


class KeyPointsDataset(BaseDataset):
    """Contains information about a keypoint (pose) dataset.

    The keypoint counterpart of
    :class:`~supervision.dataset.core.DetectionDataset`: lazy image loading,
    splitting, merging and conversion to and from the YOLO pose format.

    Each image is annotated with a
    :class:`~supervision.key_points.core.KeyPoints` instance holding one
    skeleton per object. Alongside the coordinates, the annotations carry the
    object box under ``data['xyxy']`` and, for a ``kpt_shape`` of ``[K, 3]``,
    the raw visibility values under ``data['keypoint_visibility']`` as
    ``float64``: the ``0``/``1``/``2`` flags, or the keypoint confidence that
    labels written from model output carry. Coordinates loaded by
    :meth:`~supervision.dataset.core.KeyPointsDataset.from_yolo` are ``float64``
    too, to reduce rounding during export.

    ``KeyPoints.visible`` marks every keypoint with a positive raw value: flag
    ``1`` (occluded) as well as flag ``2`` (visible), and any positive
    confidence, matching how
    :class:`~supervision.metrics.keypoint_mean_average_precision.KeyPointMeanAveragePrecision`
    and COCO evaluation count keypoints. To draw only the keypoints flagged
    ``2``, set ``key_points.visible = key_points.data['keypoint_visibility'] == 2``
    on a copy: :meth:`~supervision.dataset.core.KeyPointsDataset.as_yolo` reads
    ``visible`` as the source of truth and writes a cleared keypoint as flag
    ``0``.

    ``data['xyxy']`` is authoritative for the written box. After editing ``xy``,
    update it, or pop it to have the box derived from the labelled keypoints.

    The visibility flags are aligned with the keypoint axis rather than the
    skeleton axis. Their key is listed in
    ``supervision.config.KEYPOINT_ALIGNED_DATA_FIELDS``, so ``KeyPoints`` 2D
    indexing reorders and narrows them together with ``xy`` and ``visible``;
    every other ``data`` entry is indexed by skeleton only.

    Attributes:
        classes: List containing dataset class names.
        image_paths: List of unique image paths in insertion order; images are
            read from disk on access.
        annotations: Dictionary mapping image path to its key points.
        keypoint_count: Number of keypoints per skeleton, the ``K`` of
            ``kpt_shape``.
        value_dim: Values stored per keypoint, the ``D`` of ``kpt_shape``;
            ``2`` for coordinates only, ``3`` when a visibility flag follows.
        flip_idx: The ``flip_idx`` of the dataset as a list of Python ``int``,
            mapping each keypoint to the one it becomes under a horizontal
            flip, or ``None`` when the dataset declares none.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> ds = sv.KeyPointsDataset(
        ...     classes=['person'],
        ...     images=['img1.jpg'],
        ...     annotations={
        ...         'img1.jpg': sv.KeyPoints(
        ...             xy=np.array([[[10.0, 10.0], [20.0, 20.0]]], dtype=np.float32),
        ...             class_id=np.array([0]),
        ...         )
        ...     },
        ...     keypoint_count=2,
        ... )
        >>> len(ds)
        1

        ```
    """

    def __init__(
        self,
        classes: list[str],
        images: list[str],
        annotations: dict[str, KeyPoints],
        keypoint_count: int,
        value_dim: int = 3,
        flip_idx: Sequence[int] | npt.NDArray[np.integer] | None = None,
    ) -> None:
        """Initialize a keypoint dataset.

        Args:
            classes: Dataset class names, ordered by class index.
            images: Image paths; images are read on demand.
            annotations: Key points of each image, keyed by image path.
            keypoint_count: Number of keypoints per skeleton, a Python or NumPy
                integer stored as a Python ``int``.
            value_dim: ``2`` or ``3``, the per-keypoint value count written by
                :meth:`~supervision.dataset.core.KeyPointsDataset.as_yolo`.
            flip_idx: Optional permutation mapping each keypoint to the one it
                becomes under a horizontal flip, carried through splitting,
                merging and export. Python or NumPy integers are accepted and
                stored as a list of Python ``int``.

        Raises:
            ValueError: If the image paths and annotation keys differ, if
                ``keypoint_count`` is not a positive integer, if ``value_dim`` is
                neither 2 nor 3, if ``flip_idx`` is not a permutation of the
                keypoint indices given as integers (booleans and floats are
                refused, as the YOLO pose loader refuses them), if a non-empty
                annotation carries no ``class_id``, or if an
                annotation carries a non-integer class id, a
                class id outside the range of ``classes``, or a keypoint count
                other than ``keypoint_count``.
        """
        if set(images) != set(annotations):
            raise ValueError(
                "The keys of the images and annotations dictionaries must match."
            )
        # bool subclasses int, so `True` would otherwise pass as one keypoint.
        is_integer_count = isinstance(
            keypoint_count, (int, np.integer)
        ) and not isinstance(keypoint_count, bool)
        if not is_integer_count or keypoint_count <= 0:
            raise ValueError(
                f"`keypoint_count` must be a positive integer, got {keypoint_count!r}."
            )
        if value_dim not in (2, 3):
            raise ValueError(f"`value_dim` must be 2 or 3, got {value_dim}.")

        self.classes = classes
        # Plain ints: a NumPy scalar would be written to data.yaml as a tagged
        # object that the loader's safe YAML reader refuses.
        self.keypoint_count = int(keypoint_count)
        self._yolo_source_mapping: tuple[tuple[str, ...], tuple[int, ...]] | None = None
        self._yolo_source_paths: tuple[str, ...] = ()
        self._yolo_mapping_ambiguous = False
        self._yolo_export_class_ids: list[int] | None = None
        self._yolo_normalized_export = False
        self.value_dim = int(value_dim)
        self.flip_idx = (
            None
            if flip_idx is None
            else _validate_flip_idx(
                flip_idx=flip_idx, keypoint_count=self.keypoint_count
            )
        )
        self.annotations = {
            image_path: deepcopy(annotation)
            for image_path, annotation in annotations.items()
        }

        for image_path, annotation in self.annotations.items():
            if len(annotation) > 0 and annotation.xy.shape[1] != keypoint_count:
                raise ValueError(
                    f"Key point annotation for image {image_path!r} holds "
                    f"{annotation.xy.shape[1]} keypoints per skeleton, but the "
                    f"dataset declares {keypoint_count}."
                )
        _check_key_points_carry_class_id(annotations=self.annotations)
        _validate_and_name_class_ids(
            annotations=self.annotations, classes=self.classes, kind="Key point"
        )

        # Eliminate duplicates while preserving order
        self.image_paths = list(dict.fromkeys(images))

    def _get_image(self, image_path: str) -> npt.NDArray[np.uint8]:
        """Read an image of the dataset from disk."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from path: {image_path}")
        return cast(npt.NDArray[np.uint8], image)

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, i: int) -> tuple[str, npt.NDArray[np.uint8], KeyPoints]:
        """Load the image at index ``i`` together with its key points.

        Args:
            i: Position of the image in ``image_paths``.

        Returns:
            The image path, image data, and its key points at index i.
        """
        image_path = self.image_paths[i]
        return image_path, self._get_image(image_path), self.annotations[image_path]

    def __iter__(self) -> Iterator[tuple[str, npt.NDArray[np.uint8], KeyPoints]]:
        """Iterate over the images and annotations in the dataset.

        Yields:
            Tuples containing the image path, image data, and its key points.
        """
        for i in range(len(self)):
            yield self[i]

    def __eq__(self, other: object) -> bool:
        """Compare two keypoint datasets by content.

        Classes, image paths (order-sensitive), ``keypoint_count``,
        ``value_dim``, ``flip_idx`` and annotations must all match; image
        pixels are not read.

        Args:
            other: Object to compare against.

        Returns:
            ``True`` if ``other`` is a
            :class:`~supervision.dataset.core.KeyPointsDataset` with equal
            content, ``False`` otherwise.
        """
        if not isinstance(other, KeyPointsDataset):
            return False
        return (
            self.classes == other.classes
            and self.image_paths == other.image_paths
            and self.keypoint_count == other.keypoint_count
            and self.value_dim == other.value_dim
            and self.flip_idx == other.flip_idx
            and self.annotations == other.annotations
        )

    def split(
        self,
        split_ratio: float = 0.8,
        random_state: int | None = None,
        shuffle: bool = True,
    ) -> tuple[KeyPointsDataset, KeyPointsDataset]:
        """Split the dataset into two parts.

        The input dataset is not mutated.

        Args:
            split_ratio: Ratio of the training set to the entire dataset, within
                the inclusive range `[0, 1]`.
            random_state: Seed for the random number generator, for
                reproducibility.
            shuffle: Whether to shuffle the data before splitting.

        Returns:
            A tuple containing the training and testing datasets.

        Raises:
            ValueError: If `split_ratio` is outside `[0, 1]` or is not a finite
                number.
        """
        train_paths, test_paths = train_test_split(
            data=self.image_paths,
            train_ratio=split_ratio,
            random_state=random_state,
            shuffle=shuffle,
        )
        train_dataset, test_dataset = (
            KeyPointsDataset(
                classes=self.classes,
                images=train_paths,
                annotations={path: self.annotations[path] for path in train_paths},
                keypoint_count=self.keypoint_count,
                value_dim=self.value_dim,
                flip_idx=self.flip_idx,
            ),
            KeyPointsDataset(
                classes=self.classes,
                images=test_paths,
                annotations={path: self.annotations[path] for path in test_paths},
                keypoint_count=self.keypoint_count,
                value_dim=self.value_dim,
                flip_idx=self.flip_idx,
            ),
        )
        for subset in (train_dataset, test_dataset):
            subset._yolo_source_mapping = self._yolo_source_mapping
            subset._yolo_source_paths = self._yolo_source_paths
            subset._yolo_mapping_ambiguous = self._yolo_mapping_ambiguous
        return train_dataset, test_dataset

    @classmethod
    def merge(cls, dataset_list: list[KeyPointsDataset]) -> KeyPointsDataset:
        """Merge a list of :class:`~supervision.dataset.core.KeyPointsDataset` objects
        into one.

        Class lists are merged and each dataset's ``class_id`` values are
        remapped onto the merged list, as
        :meth:`~supervision.dataset.core.DetectionDataset.merge` does.

        Args:
            dataset_list: The datasets to merge.

        Returns:
            The merged dataset.

        Raises:
            ValueError: If the list is empty, if the datasets disagree on
                ``keypoint_count``, ``value_dim`` or ``flip_idx``, or if an
                image path appears in more than one dataset.
        """
        if not dataset_list:
            raise ValueError("Cannot merge an empty list of datasets.")

        keypoint_counts = {dataset.keypoint_count for dataset in dataset_list}
        if len(keypoint_counts) > 1:
            raise ValueError(
                "Cannot merge KeyPointsDatasets with different keypoint counts: "
                f"{sorted(keypoint_counts)}."
            )
        value_dims = {dataset.value_dim for dataset in dataset_list}
        if len(value_dims) > 1:
            raise ValueError(
                "Cannot merge KeyPointsDatasets with different `value_dim` "
                f"values: {sorted(value_dims)}."
            )
        flip_indices = {
            None if dataset.flip_idx is None else tuple(dataset.flip_idx)
            for dataset in dataset_list
        }
        if len(flip_indices) > 1:
            raise ValueError(
                "Cannot merge KeyPointsDatasets with different `flip_idx` "
                "values; they disagree on which keypoint is which: "
                f"{sorted(map(str, flip_indices))}."
            )

        image_paths = list(
            chain.from_iterable(dataset.image_paths for dataset in dataset_list)
        )
        image_paths_unique = list(dict.fromkeys(image_paths))
        if len(image_paths) != len(image_paths_unique):
            duplicates = find_duplicates(image_paths)
            raise ValueError(
                f"Image paths {duplicates} are not unique across datasets."
            )

        classes = merge_class_lists(
            class_lists=[dataset.classes for dataset in dataset_list]
        )
        annotations: dict[str, KeyPoints] = {}
        for dataset in dataset_list:
            class_index_mapping = build_class_index_mapping(
                source_classes=dataset.classes, target_classes=classes
            )
            for image_path in dataset.image_paths:
                annotation = dataset.annotations[image_path]
                # Only an empty annotation lacks class_id once constructed, and
                # it has nothing to remap; a non-empty one is rejected by cls().
                if annotation.class_id is None:
                    annotations[image_path] = annotation
                    continue
                annotations[image_path] = map_key_points_class_id(
                    source_to_target_mapping=class_index_mapping,
                    key_points=annotation,
                )

        merged_dataset = cls(
            classes=classes,
            images=image_paths_unique,
            annotations=annotations,
            keypoint_count=keypoint_counts.pop(),
            value_dim=value_dims.pop(),
            flip_idx=dataset_list[0].flip_idx,
        )
        _inherit_yolo_mapping(merged_dataset, dataset_list)
        return merged_dataset

    @classmethod
    def from_yolo(
        cls,
        images_directory_path: str,
        annotations_directory_path: str,
        data_yaml_path: str,
        show_progress: bool = False,
    ) -> KeyPointsDataset:
        """Create a dataset from a YOLO pose dataset on disk.

        The label of each object is ``class x y w h`` followed by ``K * D``
        keypoint values, with ``kpt_shape: [K, D]`` declared in data.yaml. All
        coordinates are normalized; they are denormalized against each image's
        own resolution while loading.

        Loaded ``class_id`` values index ``classes``. When ``names`` is a dict
        whose keys start above ``0`` or skip values, such as ``{1: person}``,
        each label id is mapped to the name declared for it.

        Args:
            images_directory_path: Directory holding the images.
            annotations_directory_path: Directory holding the ``.txt`` labels.
            data_yaml_path: Path to the data.yaml file, which must declare
                ``kpt_shape``.
            show_progress: If ``True``, display a progress bar while loading.

        Returns:
            The loaded dataset.

        Raises:
            FileNotFoundError: If the images or labels directory does not
                exist; an image without a label file in an existing labels
                directory still loads, with empty key points.
            NotADirectoryError: If the images or labels path names a file.
            ValueError: If data.yaml declares no valid ``kpt_shape`` or an
                invalid ``flip_idx``, declares ``names`` twice or repeats a
                class id in it, or if a label file holds an invalid line or a
                class id that ``names`` does not declare; the message names the
                file.

        Examples:
            ```python
            import supervision as sv

            ds = sv.KeyPointsDataset.from_yolo(
                images_directory_path="<IMAGES_DIRECTORY_PATH>",
                annotations_directory_path="<ANNOTATIONS_DIRECTORY_PATH>",
                data_yaml_path="<DATA_YAML_PATH>",
            )
            ```
        """
        source_data = _read_data_yaml(file_path=data_yaml_path)
        (
            classes,
            image_paths,
            annotations,
            kpt_shape,
            flip_idx,
        ) = load_yolo_pose_annotations(
            images_directory_path=images_directory_path,
            annotations_directory_path=annotations_directory_path,
            data_yaml_path=data_yaml_path,
            show_progress=show_progress,
        )
        keypoint_count, value_dim = kpt_shape
        dataset = cls(
            classes=classes,
            images=image_paths,
            annotations=annotations,
            keypoint_count=keypoint_count,
            value_dim=value_dim,
            flip_idx=flip_idx,
        )
        dataset._yolo_source_mapping = _yolo_source_mapping(
            source_data,
            _read_data_yaml(file_path=data_yaml_path),
            classes,
            data_yaml_path,
        )
        dataset._yolo_source_paths = (str(Path(data_yaml_path).resolve()),)
        return dataset

    def as_yolo(
        self,
        images_directory_path: str | None = None,
        annotations_directory_path: str | None = None,
        data_yaml_path: str | None = None,
        show_progress: bool = False,
    ) -> None:
        """Write the dataset to disk in YOLO pose format.

        The data.yaml carries ``kpt_shape: [keypoint_count, value_dim]``, so the
        written dataset reads back through
        :meth:`~supervision.dataset.core.KeyPointsDataset.from_yolo`.

        Each skeleton's box is written from ``data['xyxy']`` when present, so
        after editing ``xy`` update ``data['xyxy']`` too, or pop it to have the
        box derived from the labelled keypoints. Each visibility flag follows
        ``visible``: a cleared keypoint is written as ``0``, and a labelled one
        keeps its raw flag from ``data['keypoint_visibility']``, or ``2`` when it
        has none.

        Args:
            images_directory_path: Directory for the images; skipped when
                ``None``.
            annotations_directory_path: Directory for the ``.txt`` labels;
                skipped when ``None``.
            data_yaml_path: Path for the data.yaml file; skipped when ``None``.
                Annotation-only and YAML-only exports retain sparse source IDs;
                writing both normalizes them. If class names changed or merged
                mappings differ, export both annotations and a new YAML.
            show_progress: If ``True``, display a progress bar while saving.

        Raises:
            ValueError: Before any file is written, if two images would share an
                output file name, or if labels are requested and a non-empty
                annotation, for instance one replaced after construction, carries
                missing, misaligned or invalid class IDs, non-finite planar
                coordinates, or a ``data['xyxy']`` not of shape
                ``(N, 4)`` or a ``data['keypoint_visibility']`` not of shape
                ``(N, K)``, either holding a non-finite value, or a negative
                visibility value.

        Examples:
            ```python
            import supervision as sv

            ds.as_yolo(
                images_directory_path="<IMAGES_DIRECTORY_PATH>",
                annotations_directory_path="<ANNOTATIONS_DIRECTORY_PATH>",
                data_yaml_path="<DATA_YAML_PATH>",
            )
            ```
        """
        export_dataset = self
        if (data_yaml_path is None) != (annotations_directory_path is None):
            export_dataset = cast(KeyPointsDataset, _prepare_yolo_export(self))
        elif data_yaml_path is not None:
            export_dataset = copy(self)
            export_dataset._yolo_normalized_export = True
        # Pre-flight: validate output uniqueness before writing any file
        if images_directory_path:
            check_no_basename_collisions(
                image_paths=self.image_paths,
                key=lambda image_path: Path(image_path).name,
                output_kind="image",
            )
        if annotations_directory_path:
            check_no_basename_collisions(
                image_paths=self.image_paths,
                key=lambda image_path: Path(image_path).stem + ".txt",
                output_kind="YOLO pose annotation",
            )
        if annotations_directory_path is not None:
            # `annotations` is a public dict, so an entry may have been replaced
            # since the constructor checked it.
            _check_key_points_carry_class_id(annotations=self.annotations)
            # Checked here as well as by the label writer, because the images
            # are written first.
            _check_yolo_pose_export(dataset=self)

        if images_directory_path is not None:
            save_dataset_images(
                dataset=self,
                images_directory_path=images_directory_path,
                show_progress=show_progress,
            )
        if annotations_directory_path is not None:
            save_yolo_pose_annotations(
                dataset=export_dataset,
                annotations_directory_path=annotations_directory_path,
                show_progress=show_progress,
            )
        if data_yaml_path is not None:
            if annotations_directory_path is None and self._yolo_source_mapping:
                _save_source_yolo_yaml(self, data_yaml_path)
            else:
                save_pose_data_yaml(
                    data_yaml_path=data_yaml_path,
                    classes=self.classes,
                    keypoint_count=self.keypoint_count,
                    value_dim=self.value_dim,
                    flip_idx=self.flip_idx,
                )


@dataclass
class ClassificationDataset(BaseDataset):
    """Contains information about a classification dataset, handles lazy image loading,
    dataset splitting.

    Attributes:
        classes: List containing dataset class names.
        images:
            List of image paths or dictionary mapping image name to image data.
        annotations: Dictionary mapping
            image name to annotations.
    """

    def __init__(
        self,
        classes: list[str],
        images: list[str] | dict[str, npt.NDArray[np.uint8]],
        annotations: dict[str, Classifications],
    ) -> None:
        self.classes = classes

        if set(images) != set(annotations):
            raise ValueError(
                "The keys of the images and annotations dictionaries must match."
            )
        self.annotations = annotations

        # Eliminate duplicates while preserving order
        self.image_paths = list(dict.fromkeys(images))

        self._images_in_memory: dict[str, npt.NDArray[np.uint8]] = {}
        if isinstance(images, dict):
            self._images_in_memory = images
            warn_deprecated(
                "Passing a `Dict[str, np.ndarray]` into `ClassificationDataset` is "
                "deprecated and will be removed in a future release. Use "
                "a list of paths `List[str]` instead."
            )

    def _get_image(self, image_path: str) -> npt.NDArray[np.uint8]:
        """Assumes that image is in dataset."""
        if self._images_in_memory:
            return self._images_in_memory[image_path]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from path: {image_path}")
        return cast(npt.NDArray[np.uint8], image)

    def __len__(self) -> int:
        return len(self._images_in_memory) or len(self.image_paths)

    def __getitem__(self, i: int) -> tuple[str, npt.NDArray[np.uint8], Classifications]:
        """
        Returns:
            The image path, image data,
                and its corresponding annotation at index i.
        """
        image_path = self.image_paths[i]
        image = self._get_image(image_path)
        annotation = self.annotations[image_path]
        return image_path, image, annotation

    def __iter__(
        self,
    ) -> Iterator[tuple[str, npt.NDArray[np.uint8], Classifications]]:
        """Iterate over the images and annotations in the dataset.

        Yields:
            Tuples containing the image path, image data, and its annotation.
        """
        for i in range(len(self)):
            image_path, image, annotation = self[i]
            yield image_path, image, annotation

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClassificationDataset):
            return False

        if self.classes != other.classes:
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
        random_state: int | None = None,
        shuffle: bool = True,
    ) -> tuple[ClassificationDataset, ClassificationDataset]:
        """Splits the dataset into two parts (training and testing) using the provided
        split_ratio.

        Args:
            split_ratio: The ratio of the training set to the entire dataset,
                within the inclusive range `[0, 1]`. `0` sends every image to
                the testing dataset and `1` sends every image to the training
                dataset.
            random_state: The seed for the
                random number generator. This is used for reproducibility.
            shuffle: Whether to shuffle the data before splitting.

        Returns:
            A tuple containing
            the training and testing datasets.

        Raises:
            ValueError: If `split_ratio` is outside `[0, 1]` or is not a finite
                number.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> cd = sv.ClassificationDataset(
            ...     classes=['cat', 'dog'],
            ...     images={
            ...         'img1.jpg': np.zeros((100, 100, 3), dtype=np.uint8),
            ...         'img2.jpg': np.zeros((100, 100, 3), dtype=np.uint8),
            ...     },
            ...     annotations={
            ...         'img1.jpg': sv.Classifications(class_id=np.array([0])),
            ...         'img2.jpg': sv.Classifications(class_id=np.array([1])),
            ...     }
            ... )
            >>> train_cd, test_cd = cd.split(split_ratio=0.5, random_state=42)
            >>> len(train_cd), len(test_cd)
            (1, 1)

            ```
        """
        train_paths, test_paths = train_test_split(
            data=self.image_paths,
            train_ratio=split_ratio,
            random_state=random_state,
            shuffle=shuffle,
        )

        train_input: list[str] | dict[str, npt.NDArray[np.uint8]]
        test_input: list[str] | dict[str, npt.NDArray[np.uint8]]
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

    def as_folder_structure(
        self, root_directory_path: str, show_progress: bool = False
    ) -> None:
        """Saves the dataset as a multi-class folder structure.

        Images assigned to the same class must have unique basenames
        (case-insensitive). Conflicts are rejected before any files are written.
        Image files are copied unchanged; images held in memory are encoded to the
        format their name's extension selects.

        Args:
            root_directory_path: The path to the directory
                where the dataset will be saved.
            show_progress: If True, display a progress bar during saving.

        Raises:
            ValueError: If two images would be saved to the same output path.
            OSError: If an in-memory image cannot be written to its destination.
        """
        output_paths = {}
        for image_path in self.image_paths:
            annotation = self.annotations[image_path]
            class_id = (
                annotation.class_id[0]
                if annotation.confidence is None
                else annotation.get_top_k(1)[0][0]
            )
            output_paths[image_path] = os.path.join(
                self.classes[class_id], Path(image_path).name
            )

        check_no_basename_collisions(
            image_paths=self.image_paths,
            key=lambda path: output_paths[path],
            output_kind="classification image",
        )
        os.makedirs(root_directory_path, exist_ok=True)

        for class_name in self.classes:
            os.makedirs(os.path.join(root_directory_path, class_name), exist_ok=True)

        for image_path in tqdm(
            self.image_paths,
            desc="Saving classification images",
            disable=not show_progress,
        ):
            image_save_path = os.path.join(
                root_directory_path, output_paths[image_path]
            )
            if image_path in self._images_in_memory:
                if not cv2.imwrite(image_save_path, self._images_in_memory[image_path]):
                    raise OSError(f"Failed to save image to path: {image_save_path}")
                continue
            # Copy the file as `save_dataset_images` does: decoding and re-encoding it
            # would drop alpha and bit depth and recompress lossy formats. Exporting
            # into the folder the image was loaded from leaves it where it is.
            with contextlib.suppress(shutil.SameFileError):
                shutil.copyfile(image_path, image_save_path)

    @classmethod
    def from_folder_structure(
        cls, root_directory_path: str, show_progress: bool = False
    ) -> ClassificationDataset:
        """Load data from a multiclass folder structure into a ClassificationDataset.

        Args:
            root_directory_path: The path to the dataset directory. Hidden
                entries, root-level files, nested directories, and files whose
                suffix is not a supported image extension are ignored.
            show_progress: If True, display a progress bar during loading.

        Returns:
            The dataset.

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
                # pass show_progress=True to enable a tqdm progress bar
            )
            ```
        """
        root_directory = Path(root_directory_path)
        classes = sorted(
            {
                entry.name
                for entry in root_directory.iterdir()
                if entry.is_dir() and not entry.name.startswith(".")
            }
        )

        image_paths = []
        annotations = {}

        for class_id, class_name in enumerate(
            tqdm(
                classes,
                total=len(classes),
                desc="Loading classification dataset",
                disable=not show_progress,
            )
        ):
            class_directory = root_directory / class_name

            for image_path in sorted(
                class_directory.iterdir(), key=lambda path: path.name
            ):
                if image_path.name.startswith(".") or not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in _IMAGE_FILE_EXTENSIONS:
                    continue
                image_paths.append(str(image_path))
                annotations[str(image_path)] = Classifications(
                    class_id=np.array([class_id]),
                )

        return cls(
            classes=classes,
            images=image_paths,
            annotations=annotations,
        )
