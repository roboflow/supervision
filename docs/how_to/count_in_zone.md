With supervision, you can count the number of objects in a zone in an image or video. In this guide, we will show how to count the number of cars in a traffic video.

[View the notebook that accompanies this tutorial](https://github.com/roboflow/notebooks/blob/main/notebooks/how-to-use-polygonzone-annotate-and-supervision.ipynb).

To make it easier for you to follow our tutorial download the video we will use as an example. You can do this using supervision[assets] extension.

```python
from supervision.assets import download_assets, VideoAssets

download_assets(VideoAssets.VEHICLES_2)
```

## Initialize a Model and Load Video

First, we need to initialize a model. Let's use a YOLOv8 model with the default COCO checkpoint. We also need to load a video on which to run inference.

```python
import numpy as np
import supervision as sv
import cv2

from ultralytics import YOLO

model = YOLO('yolov8s.pt')

VIDEO = "video.mp4"

colors = sv.ColorPalette.default()
video_info = sv.VideoInfo.from_video_path(VIDEO)
```

## Calculate Coordinates

To count objects in a zone, you need to know the coordinates where you want to draw the zone.

You can calculate coordinates using the [PolygonZone web utility](https://roboflow.github.io/polygonzone/).

To use the PolygonZone website, you will need to upload an image or frame from a video. You can retrieve a frame using this code:

```python
generator = sv.get_video_frames_generator(VIDEO)
iterator = iter(generator)

frame = next(iterator)

cv2.imwrite("first_frame.png", frame)
```

PolygonZone will give you NumPy arrays that you can use with supervision to count objects in zones.

<video width="100%" loop muted autoplay>
  <source src="https://media.roboflow.com/polygonzone.mp4" type="video/mp4">
</video>

Save the coordinates in an array:

```python
polygons = [
  np.array([
    [718, 595],[927, 592],[851, 1062],[42, 1059]
  ]),
  np.array([
    [987, 595],[1199, 595],[1893, 1056],[1015, 1062]
  ])
]
```

## Define Zones

With the coordinates of the zones to draw ready, we can set up our zones:

```python
zones = [
    sv.PolygonZone(
        polygon=polygon,
        frame_resolution_wh=video_info.resolution_wh
    )
    for polygon
    in polygons
]
zone_annotators = [
    sv.PolygonZoneAnnotator(
        zone=zone,
        color=colors.by_idx(index),
        thickness=4,
        text_thickness=8,
        text_scale=4
    )
    for index, zone
    in enumerate(zones)
]
box_annotators = [
    sv.BoxAnnotator(
        color=colors.by_idx(index),
        thickness=4,
        text_thickness=4,
        text_scale=2
        )
    for index
    in range(len(polygons))
]
```

## Run Inference

We can run inference on a video using the [sv.process_video](https://supervision.roboflow.com/utils/video/#process_video) function. This function accepts a callback that runs inference on each frame and compiles the results into a video.

Below, we can call our YOLOv8 model, annotate predictions and zones, then save the results to a file called `result.mp4`.

```python
def process_frame(frame: np.ndarray, i) -> np.ndarray:
    results = model(frame, imgsz=1280, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
        mask = zone.trigger(detections=detections)
        detections_filtered = detections[mask]
        frame = box_annotator.annotate(scene=frame, detections=detections_filtered, skip_label=True)
        frame = zone_annotator.annotate(scene=frame)

    return frame

sv.process_video(source_path=VIDEO, target_path="result.mp4", callback=process_frame)
```

Here is an example of inference run on the video:

<video width="100%" loop muted autoplay>
  <source src="https://blog.roboflow.com/content/media/2023/03/trim-counting.mp4" type="video/mp4">
</video>
