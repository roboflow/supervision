With supervision, you can run inference on or otherwise process video frames with ease. In this guide, we will show how to run inference on each frame in a video and save the results to a new file.

To process video, we will use the [sv.process_video](https://supervision.roboflow.com/utils/video/#process_video) method.

## Define a Callback Function

In this guide, we will use a YOLOv8 checkpoint model to identify objects in a video. To process this video, we need to define a function that accepts a video frame and runs inference.

Create a new Python file and add the following code:

```python
import supervision as sv
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

def callback(scene: np.ndarray, index: int) -> np.ndarray:
    results = model(scene)[0]
    detections = sv.Detections.from_ultralytics(results)

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels = [
        results.names[class_id]
        for class_id
        in detections.class_id
    ]

    annotated_image = bounding_box_annotator.annotate(
        scene=scene, detections=detections)

    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    return annotated_image
```

In this file, we import the required dependencies, load a YOLOv8 model, then define a function that we will use to process video frames. This function accepts a video frame as a NumPy array, runs inference, then returns a new frame.

## Process the Video

The `sv.process_video` method lets you apply a callback function to each frame in a video and save the results to a new file. We will use the callback function we defined in the previous step to process the video.

Add the following code to your script:

```python
sv.process_video(
    source_path="video.mp4",
    target_path="output.mp4",
    callback=callback
)
```

This code accepts three arguments:

1. `source_path`: The video to process.
2. `target_path`: The path where the processed file will be saved.
3. `callback`: The function to run on each frame.

Here are the results of running our code:

<video width="100%" controls autoplay>
  <source src="https://media.roboflow.com/supervision-video-processing-example.mov" type="video/mp4">
</video>

## Low Level API

supervision provides additional utilities for use in video processing:

- [sv.get_video_frames_generator](https://supervision.roboflow.com/utils/video/#get_video_frames_generator): Create a generator that yields each frame in a video.
- [sv.VideoInfo](https://supervision.roboflow.com/utils/video/#videoinfo): Retrieve information about a video (i.e.  width, height, FPS, number of frames in the video).
- [sv.VideoSink](https://supervision.roboflow.com/utils/video/#videosink): Write frames to a video file.
