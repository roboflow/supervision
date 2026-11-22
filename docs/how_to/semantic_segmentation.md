---
comments: true
description: Load semantic segmentation results from Roboflow Inference, Ultralytics, Hugging Face Transformers or a raw class map into sv.Detections, handle the background class, read per-class confidence, annotate the masks and keep memory low with compact masks.
authors:
  - name: Borda
    role: Open Source Engineer, Roboflow
    github: https://github.com/borda
date_modified: 2026-11-22
---

# Load Semantic Segmentation Results

A semantic segmentation model labels every pixel with a class, such as *road*, *building* or *sky*, and does not separate two people standing side by side. Models return this as one dense map of shape `(H, W)` that holds a class id per pixel. supervision turns that map into [sv.Detections][supervision.detection.core.Detections] with **one detection per class present in the image**. Each detection has a mask over every pixel of its class, the tight box around that mask and the class name in `detections.data["class_name"]`.

For a side-by-side comparison of several models on one image, see the [semantic segmentation cookbook](../notebooks/semantic-segmentation.ipynb).

## Load a Result

Open the image, then load the model's result.

```python
import supervision as sv
from PIL import Image

pil_image = Image.open("<SOURCE_IMAGE_PATH>").convert("RGB")
image = sv.pillow_to_cv2(pil_image)
```

=== "Inference"

    A semantic model served by Roboflow Inference, such as an RF-DETR semantic model, returns the class map as a base64-encoded PNG together with the class names and a per-pixel confidence map. [from_inference][supervision.detection.core.Detections.from_inference] recognises the result and reads all three, the same call as for detection and instance segmentation models.

    ```python
    from inference import get_model

    model = get_model(model_id="rfdetr-sem-nano")

    detections = sv.Detections.from_inference(model.infer(pil_image)[0])
    ```

=== "Ultralytics"

    YOLO26 semantic segmentation models return the class map in `Results.semantic_mask`. [from_ultralytics][supervision.detection.core.Detections.from_ultralytics] gives one detection per class present, without confidence.

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26m-sem-ade20k.pt")

    detections = sv.Detections.from_ultralytics(model(pil_image)[0])
    ```

=== "Transformers"

    Post-process the logits into a class map at the image size, then pass it with the model's `id2label`. [from_transformers][supervision.detection.core.Detections.from_transformers] keeps every class, including class `0`.

    ```python
    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    name = "nvidia/segformer-b2-finetuned-ade-512-512"
    processor = AutoImageProcessor.from_pretrained(name)
    model = SegformerForSemanticSegmentation.from_pretrained(name)

    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    class_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[pil_image.size[::-1]]
    )[0]

    detections = sv.Detections.from_transformers(class_map, id2label=model.config.id2label)
    ```

=== "Class map"

    A raw map loads directly with [from_semantic_segmentation][supervision.detection.core.Detections.from_semantic_segmentation]. It takes an `(H, W)` integer array or tensor, or a base64-encoded PNG string without a data-URI prefix. Name the classes with `class_map`, a dict from class id to name; a class missing from it is named by its id.

    ```python
    import numpy as np

    label_map = np.load("<LABEL_MAP_NPY_PATH>")  # (H, W) integer class ids

    detections = sv.Detections.from_semantic_segmentation(
        label_map, class_map={0: "background", 1: "road", 2: "sky"}
    )
    ```

## Choose What Counts as Background

[from_semantic_segmentation][supervision.detection.core.Detections.from_semantic_segmentation] leaves out class id `0` by default, because models served by Roboflow Inference use it for background. [from_inference][supervision.detection.core.Detections.from_inference] uses the same default.

```python
# keep every class, for a model whose id 0 is a real class such as `wall`
detections = sv.Detections.from_semantic_segmentation(
    label_map, class_map=names, background_id=None
)

# leave out another id instead, such as an "unlabeled" class
detections = sv.Detections.from_semantic_segmentation(
    label_map, class_map=names, background_id=255
)
```

`from_transformers` and `from_ultralytics` keep every class. Check what id `0` means for your model before comparing results across models: a model with a background class numbers every other class one higher than a model without one.

## Read the Confidence

When the result carries a per-pixel `confidence_mask`, as an Inference result does, each detection's `confidence` is the mean of that map over the pixels of its class, divided by 255. Without a confidence map, `confidence` is `None`.

Pass a map of your own as `confidence_mask`, an 8-bit array or PNG with the same shape as the class map, to override the one in the result.

```python
if detections.confidence is not None:
    for name, confidence in zip(detections["class_name"], detections.confidence):
        print(f"{name}: {confidence:.2f}")
```

## Annotate the Result

Each detection is an ordinary row of `sv.Detections`, so the usual annotators apply. A class often wraps around other classes, so put its label at the centre of mass of its mask rather than at a corner of its box.

```python
mask_annotator = sv.MaskAnnotator(opacity=0.6)
label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER_OF_MASS)

annotated_image = mask_annotator.annotate(image.copy(), detections)
annotated_image = label_annotator.annotate(annotated_image, detections)
```

Annotators pick a colour from the class id. Two models that number the same classes differently give different colours, so pass `custom_color_lookup` with ids that agree across models to compare them.

Because the masks cover whole classes, `detections.area` is the pixel count of each class:

```python
coverage = detections.area / (image.shape[0] * image.shape[1])
```

## Keep Memory Low with Compact Masks

A dense mask array holds one `(H, W)` boolean frame per class. On a large image with many classes, pass `compact_masks=True` to get a [CompactMask][supervision.detection.compact_mask.CompactMask] instead, built one class at a time. It decodes to the same pixels as the dense array; see [Use Compact Masks](use_compact_masks.md).

```python
detections = sv.Detections.from_semantic_segmentation(
    label_map, class_map=names, background_id=None, compact_masks=True
)
```

## Workflows Output

A Roboflow Workflows semantic segmentation step does not return a raw result. At the time of writing, its output is already `sv.Detections` with `mask` set to `None` and each class's mask stored as COCO RLE in `detections.data["rle_mask"]`, so it does not go through `from_semantic_segmentation`. Calling the Inference HTTP API directly returns the raw result described above.
