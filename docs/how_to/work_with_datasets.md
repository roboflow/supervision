---
comments: true
status: new
---

# Work with Datasets

TODO

## Download Dataset from Roboflow

TODO

```python
pip install roboflow
```

```python
import roboflow

roboflow.login()
rf = roboflow.Roboflow()

workspace = rf.workspace("<WORKSPACE_ID>")
project = workspace.project("<PROJECT_ID>")
version = project.version("<DATASET_VERSION>")
dataset = version.download("<DATASET_FORMAT>")
```

We will use [football-players-detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) dataset as an example.

=== "YOLO"

    ```python
    import roboflow

    roboflow.login()
    rf = roboflow.Roboflow()
    
    workspace = rf.workspace("roboflow-jvuqo")
    project = workspace.project("football-players-detection-3zvbc")
    version = project.version(8)
    dataset = version.download("yolov8")
    ```

=== "COCO"

    ```python
    import roboflow

    roboflow.login()
    rf = roboflow.Roboflow()

    workspace = rf.workspace("roboflow-jvuqo")
    project = workspace.project("football-players-detection-3zvbc")
    version = project.version(8)
    dataset = version.download("coco")
    ```

=== "Pascal VOC"

    ```python
    import roboflow

    roboflow.login()
    rf = roboflow.Roboflow()
    
    workspace = rf.workspace("roboflow-jvuqo")
    project = workspace.project("football-players-detection-3zvbc")
    version = project.version(8)
    dataset = version.download("voc")
    ```

## Load Dataset

Lazy dataset loading is coming soon.

`ds.images` - `Dict[str, ndarray]` dictionary mapping image path to image and `ds.annotations` - `Dict[str, Detections]` dictionary mapping image name to annotations.

=== "YOLO"

    `sv.DetectionDataset.from_yolo`

    ```{ .py hl_lines="2 12-16 18" }
    import roboflow
    import supervision as sv

    roboflow.login()
    rf = roboflow.Roboflow()
    
    workspace = rf.workspace("roboflow-jvuqo")
    project = workspace.project("football-players-detection-3zvbc")
    version = project.version(8)
    dataset = version.download("yolov8")

    train_ds = sv.DetectionDataset.from_yolo(
        images_directory_path=f"{dataset.location}/train/images",
        annotations_directory_path=f"{dataset.location}/train/labels",
        data_yaml_path=f"{dataset.location}/data.yaml"
    )

    train_ds.classes
    # ['ball', 'goalkeeper', 'player', 'referee']

    len(train_ds)
    # 204
    ```

=== "COCO"

    `sv.DetectionDataset.from_coco`

    ```{ .py hl_lines="2 12-15 17" }
    import roboflow
    import supervision as sv

    roboflow.login()
    rf = roboflow.Roboflow()

    workspace = rf.workspace("roboflow-jvuqo")
    project = workspace.project("football-players-detection-3zvbc")
    version = project.version(8)
    dataset = version.download("coco")

    train_ds = sv.DetectionDataset.from_coco(
        images_directory_path=f"{dataset.location}/train",
        annotations_path=f"{dataset.location}/train/_annotations.coco.json",
    )

    train_ds.classes
    # ['ball', 'goalkeeper', 'player', 'referee']

    len(train_ds)
    # 204
    ```

=== "Pascal VOC"

    `sv.DetectionDataset.from_pascal_voc`

    ```{ .py hl_lines="2 12-15 17" }
    import roboflow
    import supervision as sv

    roboflow.login()
    rf = roboflow.Roboflow()
    
    workspace = rf.workspace("roboflow-jvuqo")
    project = workspace.project("football-players-detection-3zvbc")
    version = project.version(8)
    dataset = version.download("voc")

    train_ds = sv.DetectionDataset.from_pascal_voc(
        images_directory_path=f"{dataset.location}/train/images",
        annotations_directory_path=f"{dataset.location}/train/labels"
    )

    train_ds.classes
    # ['ball', 'goalkeeper', 'player', 'referee']

    len(train_ds)
    # 204
    ```

## Visualize Dataset

TODO

=== "YOLO"

    `sv.DetectionDataset.from_yolo`

    ```{ .py hl_lines="2 12-16 18" }
    import roboflow
    import supervision as sv

    roboflow.login()
    rf = roboflow.Roboflow()
    
    workspace = rf.workspace("roboflow-jvuqo")
    project = workspace.project("football-players-detection-3zvbc")
    version = project.version(8)
    dataset = version.download("yolov8")

    train_ds = sv.DetectionDataset.from_yolo(
        images_directory_path=f"{dataset.location}/train/images",
        annotations_directory_path=f"{dataset.location}/train/labels",
        data_yaml_path=f"{dataset.location}/data.yaml"
    )

    bounding_box_annotator = sv.BoundingBoxAnnotator()

    for image_path, image, detections in train_ds:
        pass
    ```

=== "COCO"

    `sv.DetectionDataset.from_coco`

    ```{ .py hl_lines="2 12-15 17" }
    import roboflow
    import supervision as sv

    roboflow.login()
    rf = roboflow.Roboflow()

    workspace = rf.workspace("roboflow-jvuqo")
    project = workspace.project("football-players-detection-3zvbc")
    version = project.version(8)
    dataset = version.download("coco")

    train_ds = sv.DetectionDataset.from_coco(
        images_directory_path=f"{dataset.location}/train",
        annotations_path=f"{dataset.location}/train/_annotations.coco.json",
    )

    for image_path, image, detections in train_ds:
        pass
    ```

=== "Pascal VOC"

    `sv.DetectionDataset.from_pascal_voc`

    ```{ .py hl_lines="2 12-15 17" }
    import roboflow
    import supervision as sv

    roboflow.login()
    rf = roboflow.Roboflow()
    
    workspace = rf.workspace("roboflow-jvuqo")
    project = workspace.project("football-players-detection-3zvbc")
    version = project.version(8)
    dataset = version.download("voc")

    train_ds = sv.DetectionDataset.from_pascal_voc(
        images_directory_path=f"{dataset.location}/train/images",
        annotations_directory_path=f"{dataset.location}/train/labels"
    )

    for image_path, image, detections in train_ds:
        pass
    ```

## Split Dataset

```python
import supervision as sv
```

## Merge Datasets

```python
import supervision as sv
```

## Save Dataset

=== "YOLO"

    ```python
    ```

=== "COCO"

    ```python
    ```

=== "Pascal VOC"

    ```python
    ```

## Segment Dataset

=== "YOLO"

    ```python
    ```

=== "COCO"

    ```python
    ```

=== "Pascal VOC"

    ```python
    ```