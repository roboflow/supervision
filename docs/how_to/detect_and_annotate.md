## How to Detect and Annotate

Supervision enables you to annotate predictions from object detection and segmentation models in a few lines of code.

The `sv.BoxAnnotator()` class lets you annotate images with bounding boxes, while the `sv.MaskAnnotator()` class lets you annotate images with segmentation masks. `sv.MaskAnnotator()` is a drop-in replacement for `sv.BoxAnnotator()`, so you don't need to update any other code if you want to switch between the two.

In this guide, we run inference on an Ultralytics YOLOv8 object detection model and plot the predictions with the `sv.BoxAnnotator()` class.

## Load a Model and Retrieve Predictions

First, we need to retrieve predictions from a model. For this guide, we will retrieve predictions from a YOLOv8 model. See a list of all models from which you can load predictions into supervision.

```python
import supervision as sv
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

results = model("image.jpg")
```

In this code, we import `supervision`, `ultralytics`, and `cv2` (OpenCV). We then load a YOLOv8 object detection model, then run inference on an image called `image.jpg`. The `results` variable contains the predictions from the model.

## Load Predictions into Supervision

Now that we have predictions from a model, we can load them into supervision.

We can do so using the `sv.Detections.from_ultralytics()` method, which accepts model results from Ultralytics models. See a list of other supported data loaders.

```python
detections = sv.Detections.from_ultralytics(results)
```

## Annotate Image

Next, we can annotate the image with the predictions returned by our model. Since we are working with an object detection model, we will use the `sv.BoxAnnotator()` class to annotate an image with predictions.

```python
image = cv2.imread("image.jpg")

annotator = sv.BoxAnnotator()

annotated_image = annotator.annotate(image, detections)
```

In this code, we:

1. Load the image to annotate.
2. Create a `sv.BoxAnnotator()` object to annotate the image.
3. Use the `annotate()` method to annotate the image with the predictions.

## Display Annotated Image

Finally, we can display the annotated image:

```python
sv.plot_image(annotated_image)
```

Here is the result:

![Predictions plotted on an image](https://media.roboflow.com/supervision_annotate.png)
