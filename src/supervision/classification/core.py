from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import torch


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
    if confidence is not None:
        is_valid = isinstance(confidence, np.ndarray) and confidence.shape == (n,)
        if not is_valid:
            raise ValueError("confidence must be 1d np.ndarray with (n, ) shape")


@dataclass
class Classifications:
    class_id: npt.NDArray[np.int_]
    confidence: npt.NDArray[np.floating] | None = None

    def __post_init__(self) -> None:
        """
        Validate the classification inputs.
        """
        n = len(self.class_id)

        _validate_class_ids(self.class_id, n)
        _validate_confidence(self.confidence, n)

    def __len__(self) -> int:
        """
        Returns the number of classifications.
        """
        return len(self.class_id)

    @classmethod
    def from_clip(cls, clip_results: torch.Tensor) -> Classifications:
        """
        Creates a Classifications instance from a
        [clip](https://github.com/openai/clip) inference result.

        Args:
            clip_results: The inference result from clip model.

        Returns:
            A new Classifications object.

        Example:
            ```python
            from PIL import Image
            import clip
            import supervision as sv

            model, preprocess = clip.load('ViT-B/32')

            image = cv2.imread(SOURCE_IMAGE_PATH)
            image = preprocess(image).unsqueeze(0)

            text = clip.tokenize(["a diagram", "a dog", "a cat"])
            output, _ = model(image, text)
            classifications = sv.Classifications.from_clip(output)
            ```
        """

        confidence = clip_results.softmax(dim=-1).cpu().detach().numpy()[0]

        if len(confidence) == 0:
            return cls(
                class_id=np.array([], dtype=np.int_),
                confidence=np.array([], dtype=np.float32),
            )

        class_ids = np.arange(len(confidence))
        return cls(class_id=class_ids, confidence=confidence)

    @classmethod
    def from_ultralytics(cls, ultralytics_results: Any) -> Classifications:
        """
        Creates a Classifications instance from a
        [ultralytics](https://github.com/ultralytics/ultralytics) inference result.

        Args:
            ultralytics_results: The inference result from ultralytics model.

        Returns:
            A new Classifications object.

        Example:
            ```python
            import cv2
            from ultralytics import YOLO
            import supervision as sv

            image = cv2.imread(SOURCE_IMAGE_PATH)
            model = YOLO('yolov8n-cls.pt')

            output = model(image)[0]
            classifications = sv.Classifications.from_ultralytics(output)
            ```
        """
        confidence = ultralytics_results.probs.data.cpu().numpy()
        return cls(class_id=np.arange(confidence.shape[0]), confidence=confidence)

    @classmethod
    def from_timm(cls, timm_results: Any) -> Classifications:
        """
        Creates a Classifications instance from a
        [timm](https://huggingface.co/docs/hub/timm) inference result.

        Args:
            timm_results: The inference result from timm model.

        Returns:
            A new Classifications object.

        Example:
            ```python
            from PIL import Image
            import timm
            from timm.data import resolve_data_config, create_transform
            import supervision as sv

            model = timm.create_model(
                model_name='hf-hub:nateraw/resnet50-oxford-iiit-pet',
                pretrained=True
            ).eval()

            config = resolve_data_config({}, model=model)
            transform = create_transform(**config)

            image = Image.open(SOURCE_IMAGE_PATH).convert('RGB')
            x = transform(image).unsqueeze(0)

            output = model(x)

            classifications = sv.Classifications.from_timm(output)
            ```
        """
        confidence = timm_results.cpu().detach().numpy()[0]

        if len(confidence) == 0:
            return cls(
                class_id=np.array([], dtype=np.int_),
                confidence=np.array([], dtype=np.float32),
            )

        class_id = np.arange(len(confidence))
        return cls(class_id=class_id, confidence=confidence)

    def get_top_k(
        self, k: int
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.floating]]:
        """
        Retrieve the top k class IDs and confidences,
            ordered in descending order by confidence.

        Args:
            k: The number of top class IDs and confidences to retrieve.

        Returns:
            A tuple containing the top k class IDs and confidences.

        Example:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> classifications = sv.Classifications(
            ...     class_id=np.array([0, 1, 2]),
            ...     confidence=np.array([0.3, 0.9, 0.5])
            ... )
            >>> classifications.get_top_k(1)
            (array([1]), array([0.9]))

            ```
        """
        if self.confidence is None:
            raise ValueError("top_k could not be calculated, confidence is None")

        order = np.argsort(self.confidence)[::-1]
        top_k_order = order[:k]
        top_k_class_id = self.class_id[top_k_order]
        top_k_confidence = self.confidence[top_k_order]

        return top_k_class_id, top_k_confidence
