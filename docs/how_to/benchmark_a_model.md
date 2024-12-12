---
comments: true
status: new
---

<!-- TODO: replace all images -->

<!-- ![Corgi Example](https://media.roboflow.com/supervision/image-examples/how-to/benchmark-models/corgi-sorted-2.png) -->

# Benchmark a Model

This guide shows how to benchmark a model on a test dataset, visualizing the results with [`ComparisonAnnotator`](https://supervision.roboflow.com/develop/detection/annotators/#supervision.annotators.core.ComparisonAnnotator), computing [`MeanAveragePrecision`](https://supervision.roboflow.com/latest/metrics/mean_average_precision/#supervision.metrics.mean_average_precision.MeanAveragePrecision) and [`F1Score`](https://supervision.roboflow.com/latest/metrics/f1_score/). This will allow you to understand how well a model performs on a given task.

!!! tip

    This guide is also available as a [Colab Notebook](https://colab.research.google.com/drive/1HoOY9pZoVwGiRMmLHtir0qT6Uj45w6Ps?usp=sharing).

We'll use the following libraries:

- [`roboflow`](https://github.com/roboflow/roboflow-python) to manage the dataset and deploy models
- [`inference`](https://github.com/roboflow/inference) to run the models
- [`supervision`](https://github.com/roboflow/supervision) to evaluate the model results

```bash
pip install roboflow supervision
pip install git+https://github.com/roboflow/inference.git@linas/allow-latest-rc-supervision
```

!!! info

    We're updating `inference` at the moment. Please use the installation method above.

## Benchmarking Basics

<!-- TODO: upload to marketing -->

<!-- ![dataset-splits](dataset-splits.png) -->

First you need to select a set of images to benchmark on. Datasets are frequently split into `train`, `validation` and `testing` subsets. Here are the situations you might encounter:

- **Unrelated Dataset**: You may have a dataset that wasn't used to train the model. This is the best choice for benchmarking, and any image from it can be used.
- **Training Set**: This is the set of images used to train the model. _Never_ use it for benchmarking - the results will seem unrealistically good.
- **Validation Set**: This is the set of images used to validate the model during training. Every Nth training epoch, the model is evaluated on the validation set. Often training is stopped once the validation stops showing improvement. Therefore, while the images aren't used to train the model, it indirectly influences the training outcome.
- **Test Set**: These images were kept aside for testing the model. Use these for benchmarking.

Therefore, an unrelated dataset or the `test` set is the best choice for model evaluation.

Note several data-related issues that may arise:

- **Extra Classes**: A dataset may contain some classes that a model wasn't trained to recognize. You need to [filter out](https://supervision.roboflow.com/how_to/filter_detections/#by-set-of-classes) these before computing metrics.
- **Class Mismatch**: The class names or IDs may be different to what your model produces. This guide includes a section on how to [remap classes](#remap-classes).
- **Data Contamination**: The `test` set may not be split correctly, with images from the test set also present in `training` or `validation` set and used during training. In this case, the results will be overly optimistic. This also applies when _very similar_ images are used for training and testing - e.g. those taken in the same environment, same lighting conditions, similar camera angles, etc.
- **Missing Test Set**: Some datasets do not come with a test set. In this case, you should collect and [label](https://roboflow.com/annotate) your own data. Alternatively, a validation set could be used, but the results could be overly optimistic. Make sure to test in the real world as soon as possible.

## Download a Dataset

This guide will use the [Corgi v2](https://universe.roboflow.com/fbamse1-gm2os/corgi-v2) dataset from [Roboflow Universe](https://universe.roboflow.com/).

!!! tip

    To learn how to label your own datasets, see this [guide](https://roboflow.com/how-to-label/yolo11).

=== "YOLO"

    ```python
    import roboflow

    roboflow.login()

    rf = roboflow.Roboflow()
    project = rf.workspace("fbamse1-gm2os").project("corgi-v2")
    dataset = project.version(4).download("yolov11")
    ```

=== "COCO"

    ```python
    import roboflow

    roboflow.login()

    rf = roboflow.Roboflow()
    project = rf.workspace("fbamse1-gm2os").project("corgi-v2")
    dataset = project.version(4).download("coco")
    ```

=== "Pascal VOC"

    ```python
    import roboflow

    roboflow.login()

    rf = roboflow.Roboflow()
    project = rf.workspace("fbamse1-gm2os").project("corgi-v2")
    dataset = project.version(4).download("voc")
    ```

This will create a folder called `Corgi-v2-4` with the dataset in the current working directory.

## Load the Test Set

`supervision`provides tools to work with many [Dataset](https://supervision.roboflow.com/latest/datasets/core/) formats. This will make it easy to iterate over the test set images and annotations, without loading the entire dataset into memory.

=== "YOLO"

    ```python
    from supervision import DetectionDataset

    test_set = DetectionDataset.from_yolo(
        images_directory_path=f"{dataset.location}/test/images",
        annotations_directory_path=f"{dataset.location}/test/labels",
        data_yaml_path=f"{dataset.location}/data.yaml"
    )
    ```

=== "COCO"

    ```python
    from supervision import DetectionDataset

    test_set = DetectionDataset.from_coco(
        images_directory_path=f"{dataset.location}/test",
        annotations_path=f"{dataset.location}/test/_annotations.coco.json",
    )
    ```

=== "Pascal VOC"

    ```python
    from supervision import DetectionDataset

    test_set = DetectionDataset.from_voc(
        images_directory_path=f"{dataset.location}/test/images",
        annotations_directory_path=f"{dataset.location}/test/labels"
    )
    ```

## Load a Model

First, you'll need to load a model from the library of your choice. `supervision` supports [more than 15](https://supervision.roboflow.com/latest/detection/core/#supervision.detection.core.Detections-functions) popular model runtimes, accessible with `from_X` functions.

=== "Inference"

    ```python
    from inference import get_model

    model = get_model(model_id="yolov11s-640")
    ```

    !!! tip

        A full list of pre-trained models IDs is available in [inference documentation](https://inference.roboflow.com/quickstart/aliases/).

=== "Ultralytics"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11s.pt")
    ```

## Run the Model

Next, run the model on each test image, aggregating the results.

=== "Inference"

    ```python
    import supervision as sv

    image_paths = []
    predictions_list = []
    targets_list = []

    for image_path, image, label in test_set:
        result = model.infer(image)[0]
        predictions = sv.Detections.from_inference(result)

        image_paths.append(image_path)
        predictions_list.append(predictions)
        targets_list.append(label)
    ```

=== "Ultralytics"

    ```python
    import supervision as sv

    image_paths = []
    predictions_list = []
    targets_list = []

    for image_path, image, label in test_set:
        result = model(image)[0]
        predictions = sv.Detections.from_ultralytics(result)

        image_paths.append(image_path)
        predictions_list.append(predictions)
        targets_list.append(label)
    ```

## Remap classes

If the model was trained on a different dataset, you may find that class IDs and names of predictions differ from the dataset classes. It is important to unify these.

```python
def remap_classes(
    detections: sv.Detections,
    class_ids_from_to: dict[int, int],
    class_names_from_to: dict[str, str]
) -> None:
    new_class_ids = [
        class_ids_from_to.get(class_id, class_id) for class_id in detections.class_id]
    detections.class_id = np.array(new_class_ids)

    new_class_names = [
        class_names_from_to.get(name, name) for name in detections["class_name"]]
    predictions["class_name"] = np.array(new_class_names)
```

Additionally, let's remove any predictions that aren't part of the test set.

=== "Inference"

    Dataset class names and IDs can be found by printing `dataset.classes`, or in the `data.yaml` file.

    ```python
    import supervision as sv

    image_paths = []
    predictions_list = []
    targets_list = []

    for image_path, image, label in test_set:
        result = model.infer(image)[0]
        predictions = sv.Detections.from_inference(result)

        remap_classes(
            detections=predictions,
            class_ids_from_to={16: 0},
            class_names_from_to={"dog": "Corgi"}
        )
        predictions = predictions[
            np.isin(predictions["class_name"], test_set.classes)
        ]

        image_paths.append(image_path)
        predictions_list.append(predictions)
        targets_list.append(label)
    ```

=== "Ultralytics"

    Dataset class names and IDs can be found in the `data.yaml` file, or by printing `dataset.classes`.

    Each model will have a different class mapping, so make sure to check the model's [documentation](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml).

    ```python
    import supervision as sv

    image_paths = []
    predictions_list = []
    targets_list = []

    for image_path, image, label in test_set:
        result = model(image)[0]
        predictions = sv.Detections.from_ultralytics(result)

        remap_classes(
            detections=predictions,
            class_ids_from_to={16: 0},
            class_names_from_to={"dog": "Corgi"}
        )
        predictions = predictions[
            np.isin(predictions["class_name"], test_set.classes)
        ]

        image_paths.append(image_path)
        predictions_list.append(predictions)
        targets_list.append(label)
    ```

## Visualize Predictions

The first step in evaluating your model’s performance is to visualize its predictions.
This gives an intuitive sense of how well your model is detecting objects and where it might be failing. We can use [`ComparisonAnnotator`](https://supervision.roboflow.com/develop/detection/annotators/#supervision.annotators.core.ComparisonAnnotator) for this task.

```python
import supervision as sv

N = 9
GRID_SIZE = (3, 3)

comparison_annotator = sv.ComparisonAnnotator(
    label_1="Predictions",
    label_2="Targets",
    label_overlap="Overlap",
)

annotated_images = []
for image_path, predictions, targets in zip(
    image_paths[:N], predictions_list[:N], targets_list[:N]
):
    annotated_image = cv2.imread(image_path)
    annotated_image = comparison_annotator.annotate(
        scene=annotated_image,
        detections_1=predictions,
        detections_2=targets
    )
    annotated_images.append(annotated_image)

sv.plot_images_grid(images=annotated_images, grid_size=GRID_SIZE)
```

Here, the purple areas are targets (ground truth), and red areas are predictions. Green represents the areas where detections match the targets. The larger the green overlap, and the smaller the purple and red areas, the better the model performs.

<!-- ![Basic Model Comparison](https://media.roboflow.com/supervision/image-examples/how-to/benchmark-models/basic-model-comparison-corgi.png) -->

!!! tip

    The [`ComparisonAnnotator`](https://supervision.roboflow.com/develop/detection/annotators/#supervision.annotators.core.ComparisonAnnotator) supports object detection, instance segmentation, and OBB!

## Benchmark with Metrics

`supervision` provides a collection of metrics that help obtain precise numerical results of model performance. In this section we'll look at [MeanAveragePrecision (mAP)](https://supervision.roboflow.com/latest/metrics/mean_average_precision/#supervision.metrics.mean_average_precision.MeanAveragePrecision) and [F1 Score](https://supervision.roboflow.com/latest/metrics/f1_score/).

### Mean Average Precision (mAP)

[MeanAveragePrecision (mAP)](https://supervision.roboflow.com/latest/metrics/mean_average_precision/#supervision.metrics.mean_average_precision.MeanAveragePrecision) is one of the most common metric for measuring object detection results. It computes the average precision across classes, recall levels and IoU thresholds, giving a single number that summarizes the model's performance.

The primary value is `mAP 50:95`. It represents the average precision across all classes and IoU thresholds (`0.5` to `0.95`), whereas other values such as `mAP 50` or `mAP 75` only consider a single IoU threshold (`0.5` and `0.75` respectively).

The metric breaks down the results by detected object area. Small, medium and large labels correspond to objects with area less than 32², between 32² and 96², and greater than 96² pixels respectively.

!!! info

    For a thorough explanation of the metric, check out our [blog](https://blog.roboflow.com/mean-average-precision/) and [Youtube video](https://www.youtube.com/watch?v=oqXDdxF_Wuw).

Let's compute the mAP:

```python
from supervision.metrics import MeanAveragePrecision, MetricTarget

map_metric = MeanAveragePrecision()
map_result = map_metric.update(predictions_list, targets_list).compute()

print(map_result)
```

```
MeanAveragePrecisionResult:
Metric target: MetricTarget.MASKS
Class agnostic: False
mAP @ 50:95: 0.2409
mAP @ 50:    0.3591
mAP @ 75:    0.2915
mAP scores: [0.35909 0.3468 0.34556 ...]
IoU thresh: [0.5 0.55 0.6 ...]
AP per class:
  0: [0.35909 0.3468 0.34556 ...]
...
Small objects: ...
Medium objects: ...
Large objects: ...
```

You can also plot the results:

```python
map_result.plot()
```

<!-- ![mAP Plot](https://media.roboflow.com/supervision/image-examples/how-to/benchmark-models/mAP-plot-corgi.png) -->

### F1 Score

The [F1 Score](https://supervision.roboflow.com/latest/metrics/f1_score/) is another useful metric, especially when dealing with an imbalance between false positive and false negative cases. It’s the harmonic mean of **precision** (how many predictions are correct) and **recall** (how many actual instances were detected).

The metric breaks down the results by detected object area. Small, medium and large labels correspond to objects with area less than 32², between 32² and 96², and greater than 96² pixels respectively.

Here's how you can compute the F1 score:

```python
from supervision.metrics import F1Score, MetricTarget

f1_metric = F1Score()
f1_result = f1_metric.update(predictions_list, targets_list).compute()

print(f1_result)
```

```
F1ScoreResult:
Metric target: MetricTarget.MASKS
Averaging method: AveragingMethod.WEIGHTED
F1 @ 50:     0.5341
F1 @ 75:     0.4636
F1 @ thresh: [0.53406 0.5278 0.52153 ...]
IoU thresh: [0.5 0.55 0.6 ...]
F1 per class:
  0: [0.53406 0.5278 0.52153 ...]
...
Small objects: ...
Medium objects: ...
Large objects: ...
```

You can also plot the results:

```python
f1_result.plot()
```

<!-- ![F1 Plot](https://media.roboflow.com/supervision/image-examples/how-to/benchmark-models/f1-score-corgi.png) -->

## Model Leaderboard

We have carried out benchmarking on a wide range of models. Check out our [Model Leaderboard](https://leaderboard.roboflow.com/) to see how different models perform and to get a sense of the state-of-the-art results. It's a great place to understand what the leading models can achieve and to compare your own results.

The repository is open source. You can see how the models were benchmarked, run the evaluation yourself, and even add your own models to the leaderboard. Check it out on [GitHub](https://github.com/roboflow/model-leaderboard)!

![Model Leaderboard Example](https://media.roboflow.com/model-leaderboard/model-leaderboard-example.png)

## Summary

This guide has explained the steps on how to benchmark a model. It has shown how to setup the environment, use pre-trained models, visualize predictions with [`ComparisonAnnotator`](https://supervision.roboflow.com/develop/detection/annotators/#supervision.annotators.core.ComparisonAnnotator), and evaluate model performance with [`mAP`](https://supervision.roboflow.com/latest/metrics/mean_average_precision/) and [`F1 score`](https://supervision.roboflow.com/latest/metrics/f1_score/).

For more details, be sure to check out the [Colab Notebook](https://colab.research.google.com/drive/1HoOY9pZoVwGiRMmLHtir0qT6Uj45w6Ps?usp=sharing), our [documentation](https://supervision.roboflow.com/latest/) the [discussion board](https://discuss.roboflow.com/). If you find any issues, please let us know on [GitHub](https://github.com/roboflow/supervision/issues).
