from typing import Dict, List

import numpy as np

from supervision import DetectionDataset, Detections


def mock_detections(
    xyxy: List[List[float]],
    confidence: List[float] = None,
    class_id: List[int] = None,
    tracker_id: List[int] = None
) -> Detections:
    return Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=confidence if confidence is None else np.array(confidence, dtype=np.float32),
        class_id=class_id if class_id is None else np.array(class_id, dtype=int),
        tracker_id=tracker_id if tracker_id is None else np.array(tracker_id, dtype=int)
    )


def mock_detection_dataset(
    images: Dict[str, np.ndarray],
    annotations: Dict[str, Detections],
    classes: List[str],
) -> DetectionDataset:
    return DetectionDataset(classes=classes, images=images, annotations=annotations)


def dummy_detection_dataset():
    img_paths = ["a.png", "b.png", "c.png"]
    classes = ["a", "b", "c"]
    imgs = [
        np.random.randint(0, 255, size=(28, 28, 3), dtype=np.uint8),
        np.random.randint(0, 255, size=(28, 28, 3), dtype=np.uint8),
        np.random.randint(0, 255, size=(28, 28, 3), dtype=np.uint8),
    ]
    detections = [
        mock_detections(
            xyxy=[[10, 10, 20, 20], [20, 20, 25, 25]],
            class_id=[0, 1],
            confidence=np.ones(2),
        ),
        mock_detections(
            xyxy=[[10, 10, 20, 20], [20, 20, 25, 25]],
            class_id=[2, 1],
            confidence=np.ones(2),
        ),
        mock_detections(
            xyxy=[[10, 10, 20, 20], [20, 20, 25, 25], [10, 10, 15, 15]],
            class_id=[0, 2, 2],
            confidence=np.ones(3),
        ),
    ]
    annotations = dict(zip(img_paths, detections))
    images = dict(zip(img_paths, imgs))
    dataset = mock_detection_dataset(images, annotations, classes)
    return dataset


def dummy_detection_dataset_with_map_img_to_annotation():
    dataset = dummy_detection_dataset()
    # dataset.img2ann = {
    #     tuple(img): dataset.annotations[img_key] for img_key, img in dataset.images.items()
    # }
    dataset.map_img_to_annotation = lambda img: dataset.annotations[
        [k for k, v in dataset.images.items() if np.array_equal(v, img)][0]
    ]
    return dataset
