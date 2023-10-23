supervision enables you to both process detections from a model and datasets. Dataset processing is implemented in the `sv.DetectionDataset` (object detection and segmentation) and `sv.ClassificationDataset` (classification) APIs.

The supervision `sv.DetectionDataset` and `sv.ClassificationDataset` APIs enables you to: 

1. Load full datasets into supervision
2. Split datasets into train/test sets
3. Merge datasets together

Each image in a `DetectionDataset` object is assigned an [sv.Detections](https://supervision.roboflow.com/detection/core/) object that you can manipulate. Each image in a `ClassificationDataset` object is assigned a [Classifications](https://supervision.roboflow.com/classification/core/) object that you can manipulate.

In this guide, we will walk through how to accomplish all of the above tasks in supervision.

## Processing Detection Datasets

### Load a Dataset into Supervision

To load a dataset into supervision, you need to use a data loader. For this guide, we will load a COCO dataset, so we will use the `DetectionDataset.from_coco` data loader.

The following data loaders are supported:

- `DetectionDataset.from_coco` ([COCO JSON](https://roboflow.com/formats/coco-json))
- `DetectionDataset.from_yolo` ([YOLO PyTorch TXT](https://roboflow.com/formats/yolov8-pytorch-txt))
- `DetectionDataset.from_pascal_voc` ([Pascal VOC XML](https://roboflow.com/formats/pascal-voc-xml))

Create a new Python file and add the following code:

```python
import supervision as sv

DATASET_PATH = "football-players-detection"

ds = sv.DetectionDataset.from_yolo(
    images_directory_path=f"{DATASET_PATH}/train/images",
    annotations_directory_path=f"{DATASET_PATH}/train/labels",
    data_yaml_path=f"{DATASET_PATH}/data.yaml"
)

print(ds.classes)
# ['ball', 'goalkeeper', 'player', 'referee']
```

This code loads a dataset stored in the YOLOv8 PyTorch TXT format into an `sv.DetectionDataset` object. Then, the classes in the dataset are printed out to the console.

### Split a Dataset into Train/Test Sets

To split a dataset into train/test datasets, you can use the `sv.DetectionDataset.split` method.

```python
train_ds, test_ds = ds.split(
    split_ratio=0.7,
    random_state=42,
    shuffle=True
)
```

This code creates two `sv.DetectionDataset` instances. The first contains a train dataset and the second contains the test dataset. We have specified a 0.7 split, which means 70% of images will go to the test set.

You can use `random_state` to set a seed you can use to reproduce the same split. You can use `shuffle` to shuffle the dataset before splitting.

### Visualize Annotations

You can visualize annotations from an object detection and segmentation dataset using the `sv.BoundingBoxAnnotator` and `sv.MaskAnnotator` methods. See documentation for supervision anontators.

Let's visualize an image in a object detection dataset.

```python
image_name = DATASET_PATH + "/train/images/42ba34_9_9_png.rf.1f36573ac36d8b56c1f0a2f11bd480d4.jpg"

image = ds.images[image_name]
annotations = ds.annotations[image_name]

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [
    ds.classes[class_id]
    for class_id
    in annotations.class_id
]

annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=annotations)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=annotations, labels=labels)

sv.plot_image(annotated_image)
```

Here is the output:

![Annotated Image of players on a football pitch](https://media.roboflow.com/football-players-supervision-example.png)

In the code above, we use retrieve an image from the dataset through the `ds.images` dictionary and its associated annotations (represented as a `sv.Detections` object) through the `ds.annotations` dictionary.

We use the `sv.BoundingBoxAnnotator` and `sv.LabelAnnotator` to annotate the image with bounding boxes and labels. We then plot the image using the `sv.plot_image` method.

## Merge Datasets

You can merge two detection datasets together using the `sv.DetectionDataset.merge` method.

```python
merged_ds = sv.ClassificationDataset.merge(
    [cd_train, cd_test]
)
```

## Processing Classification Datasets

You can work with classification datasets using the `sv.ClassificationDataset` API.

### Load a Dataset into Supervision

To load a dataset into supervision, you need to use a data loader. You can load detections from a classification dataset using the `sv.ClassificationDataset.from_folder_structure` data loader.

```python
import supervision as sv

cd_train = sv.ClassificationDataset.from_folder_structure(
    "artwork/train"
)
cd_test = sv.ClassificationDataset.from_folder_structure(
    "artwork/test"
)
cd_valid = sv.ClassificationDataset.from_folder_structure(
    "artwork/valid"
)

print(cd_train.classes)
# ['abstract', 'abstract digital', 'abstract digital landscape surrealism', 'abstract digital surrealism', ...]
```

`dataset/` is the path where your classification folder dataset is stored.

### Split a Dataset into Train/Test Sets

To split a dataset into train/test datasets, you can use the `sv.ClassificationDataset.split` method.

```python
train_ds, test_ds = ds.split(
    split_ratio=0.7,
    random_state=42,
    shuffle=True
)
```

This code creates two `sv.ClassificationDataset` instances. The first contains a train dataset and the second contains the test dataset. We have specified a 0.7 split, which means 70% of images will go to the test set.

### Retrieve Annotations

You can retrieve annotations from a classification dataset using the `sv.ClassificationDataset.annotations` dictionary.

```python
image = "artwork/train/abstract digital/03c8e4c4430d631029694e64f4d29b97_jpg.rf.378dcf12b0adb97ee966d84fb63a0e28.jpg"

classes = cd_train.classes

print(classes[cd_train.annotations[image].class_id[0]])
# ['abstract digital']
```