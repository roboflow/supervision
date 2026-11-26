---
comments: true
description: Load image classification results from Ultralytics, timm, CLIP, or a raw score array into sv.Classifications, read the top classes with their names, draw the label on the image, load, split, and save classification datasets.
authors:
  - name: jmurel
    role: Community Contributor
    github: https://github.com/jmurel
date_modified: 2026-11-25
---

# Classify Images

An image classification model scores a whole image against a set of classes, such as *cat* or *dog*. Unlike object detection or segmentation, it returns no boxes or masks. `supervision` loads these scores into [sv.Classifications][supervision.classification.core.Classifications], which holds a `class_id` integer and a `confidence` score for every class. It stores no class names, so each example below keeps them in `class_names`.

## Load a Result

Open the image, then load the model's result.

```python
import supervision as sv
from PIL import Image

pil_image = Image.open("<SRC_IMAGE_PATH>").convert("RGB")
image = sv.pillow_to_cv2(pil_image)
```

### Ultralytics

Ultralytics classification models return class probabilities and class names respectively in the forms `Results.probs` and and `Results.names`. Use `sv.Classifications.from_ultralytics` to read the probabilities.

```python
from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")
result = model(pil_image)[0]

classifications = sv.Classifications.from_ultralytics(result)
class_names = result.names
```

### timm

timm models return raw logits. Use `sv.Classifications.from_timm` to apply softmax to logits so confidence scores sum to 1.

```python
import timm
import torch
from timm.data import create_transform, resolve_data_config

model = timm.create_model(
    "hf-hub:nateraw/resnet50-oxford-iiit-pet", pretrained=True
).eval()
transform = create_transform(**resolve_data_config({}, model=model))

with torch.no_grad():
    output = model(transform(pil_image).unsqueeze(0))

classifications = sv.Classifications.from_timm(output)
class_names = model.pretrained_cfg["label_names"]
```

### CLIP

CLIP scores an image against text prompts, with prompts being class names. Much like `from_timm`, use `sv.Classifications.from_clip` to apply softmax to image logits.

```python
import clip
import torch

model, preprocess = clip.load("ViT-B/32", device="cpu")
class_names = ["a diagram", "a dog", "a cat"]

with torch.no_grad():
    logits_per_image, _ = model(
        preprocess(pil_image).unsqueeze(0), clip.tokenize(class_names)
    )

classifications = sv.Classifications.from_clip(logits_per_image)
```

### Other models

For a model without a connector, you need to pass one score per class as a NumPy array.

```python
import numpy as np

scores = np.array([0.1, 0.7, 0.2])
class_names = ["cat", "dog", "fox"]

classifications = sv.Classifications(class_id=np.arange(len(scores)), confidence=scores)
```

## Read Top Classes

The get `Classifications.get_top_k` method returns the `k` most confident class ids alongside their corresponding confidence scores, highest first. You may use a for loop to look up each name by its id.

```python
class_ids, confidences = classifications.get_top_k(3)

for class_id, confidence in zip(class_ids, confidences):
    print(f"{class_names[class_id]}: {confidence:.2f}")
```

## Draw Labels

No annotator takes `sv.Classifications`. Write the label on the image with the `draw_text()` function, which centers the text on `text_anchor` argument value.

```python
label = f"{class_names[class_ids[0]]} {confidences[0]:.2f}"

annotated_image = sv.draw_text(
    scene=image.copy(),
    text=label,
    text_anchor=sv.Point(x=image.shape[1] // 2, y=30),
    text_color=sv.Color.WHITE,
    background_color=sv.Color.BLACK,
)
```

## Load Dataset

`ClassificationDataset` reads a dataset stored as one folder per class. The `from_folder_structure()` function uses folder names as class names in sorted order. It loads images only when they are read.

```python
ds = sv.ClassificationDataset.from_folder_structure(
    root_directory_path="<DATASET_DIR_PATH>"
)

for image_path, image, annotation in ds:
    print(image_path, ds.classes[annotation.class_id[0]])
```

Each annotation is a `sv.Classifications` with one class id and no confidence. For detection and segmentation datasets, see [Process Datasets](process_datasets.md).

## Split and Save Dataset

The `split()` function divides images at random, so two parts can hold different shares of each class. `as_folder_structure()` writes each part back as one folder per class.

```python
ds_train, ds_test = ds.split(split_ratio=0.8, random_state=42)

ds_train.as_folder_structure(root_directory_path="<TRAIN_DIR_PATH>")
ds_test.as_folder_structure(root_directory_path="<TEST_DIR_PATH>")
```

## Sort Images by Predicted Class

`as_folder_structure` files an image under its most confident class when the annotation has confidences. Build a dataset from predictions to sort unlabeled images into class folders. A folder is created for every class, including classes to which no image was assigned. This example uses `model` from the **Ultralytics** tab.

```python
image_paths = sv.list_files_with_extensions(
    directory="<IMAGES_DIR_PATH>", extensions=["jpg", "png"]
)

predictions = {}
for image_path in image_paths:
    result = model(str(image_path))[0]
    predictions[str(image_path)] = sv.Classifications.from_ultralytics(result)

ds = sv.ClassificationDataset(
    classes=list(model.names.values()),
    images=list(predictions),
    annotations=predictions,
)
ds.as_folder_structure(root_directory_path="<SORTED_DIR_PATH>")
```
