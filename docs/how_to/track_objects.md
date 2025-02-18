---
comments: true
status: new
---

# Track Objects

Leverage Supervision's advanced capabilities for enhancing your video analysis by
seamlessly [tracking](/latest/trackers/) objects recognized by
a multitude of object detection, segmentation and keypoint models. This comprehensive guide will
take you through the steps to perform inference using the YOLOv8 model via either the
[Inference](https://github.com/roboflow/inference) or
[Ultralytics](https://github.com/ultralytics/ultralytics) packages. Following this,
you'll discover how to track these objects efficiently and annotate your video content
for a deeper analysis.

## Object Detection & Segmentation

To make it easier for you to follow our tutorial download the video we will use as an
example. You can do this using
[`supervision[assets]`](/latest/assets/) extension.

```python
from supervision.assets import download_assets, VideoAssets

download_assets(VideoAssets.PEOPLE_WALKING)
```

<video controls>
    <source src="https://media.roboflow.com/supervision/video-examples/people-walking.mp4" type="video/mp4">
</video>

### Run Inference

First, you'll need to obtain predictions from your object detection or segmentation
model. In this tutorial, we are using the YOLOv8 model as an example. However,
Supervision is versatile and compatible with various models. Check this
[link](/latest/how_to/detect_and_annotate/#load-predictions-into-supervision)
for guidance on how to plug in other models.

We will define a `callback` function, which will process each frame of the video
by obtaining model predictions and then annotating the frame based on these predictions.
This `callback` function will be essential in the subsequent steps of the tutorial, as
it will be modified to include tracking, labeling, and trace annotations.

!!! tip

    Both object detection and segmentation models are supported. Try it with `yolov8n.pt` or `yolov8n-640-seg`!

=== "Ultralytics"

    ```{ .py }
    import numpy as np
    import supervision as sv
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    box_annotator = sv.BoxAnnotator()

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        return box_annotator.annotate(frame.copy(), detections=detections)

    sv.process_video(
        source_path="people-walking.mp4",
        target_path="result.mp4",
        callback=callback
    )
    ```

=== "Inference"

    ```{ .py }
    import numpy as np
    import supervision as sv
    from inference.models.utils import get_roboflow_model

    model = get_roboflow_model(model_id="yolov8n-640", api_key=<ROBOFLOW API KEY>)
    box_annotator = sv.BoxAnnotator()

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)
        return box_annotator.annotate(frame.copy(), detections=detections)

    sv.process_video(
        source_path="people-walking.mp4",
        target_path="result.mp4",
        callback=callback
    )
    ```

<video controls>
    <source src="https://media.roboflow.com/supervision/video-examples/how-to/track-objects/run-inference.mp4" type="video/mp4">
</video>

### Tracking

After running inference and obtaining predictions, the next step is to track the
detected objects throughout the video. Utilizing Supervision’s
[`sv.ByteTrack`](/latest/trackers/#supervision.tracker.byte_tracker.core.ByteTrack)
functionality, each detected object is assigned a unique tracker ID,
enabling the continuous following of the object's motion path across different frames.

=== "Ultralytics"

    ```{ .py hl_lines="6 12" }
    import numpy as np
    import supervision as sv
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)
        return box_annotator.annotate(frame.copy(), detections=detections)

    sv.process_video(
        source_path="people-walking.mp4",
        target_path="result.mp4",
        callback=callback
    )
    ```

=== "Inference"

    ```{ .py hl_lines="6 12" }
    import numpy as np
    import supervision as sv
    from inference.models.utils import get_roboflow_model

    model = get_roboflow_model(model_id="yolov8n-640", api_key=<ROBOFLOW API KEY>)
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)
        detections = tracker.update_with_detections(detections)
        return box_annotator.annotate(frame.copy(), detections=detections)

    sv.process_video(
        source_path="people-walking.mp4",
        target_path="result.mp4",
        callback=callback
    )
    ```

### Annotate Video with Tracking IDs

Annotating the video with tracking IDs helps in distinguishing and following each object
distinctly. With the
[`sv.LabelAnnotator`](/latest/detection/annotators/#supervision.annotators.core.LabelAnnotator)
in Supervision, we can overlay the tracker IDs and class labels on the detected objects,
offering a clear visual representation of each object's class and unique identifier.

=== "Ultralytics"

    ```{ .py hl_lines="8 15-19 23-24" }
    import numpy as np
    import supervision as sv
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        labels = [
            f"#{tracker_id} {results.names[class_id]}"
            for class_id, tracker_id
            in zip(detections.class_id, detections.tracker_id)
        ]

        annotated_frame = box_annotator.annotate(
            frame.copy(), detections=detections)
        return label_annotator.annotate(
            annotated_frame, detections=detections, labels=labels)

    sv.process_video(
        source_path="people-walking.mp4",
        target_path="result.mp4",
        callback=callback
    )
    ```

=== "Inference"

    ```{ .py hl_lines="8 15-19 23-24" }
    import numpy as np
    import supervision as sv
    from inference.models.utils import get_roboflow_model

    model = get_roboflow_model(model_id="yolov8n-640", api_key=<ROBOFLOW API KEY>)
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)
        detections = tracker.update_with_detections(detections)

        labels = [
            f"#{tracker_id} {results.names[class_id]}"
            for class_id, tracker_id
            in zip(detections.class_id, detections.tracker_id)
        ]

        annotated_frame = box_annotator.annotate(
            frame.copy(), detections=detections)
        return label_annotator.annotate(
            annotated_frame, detections=detections, labels=labels)

    sv.process_video(
        source_path="people-walking.mp4",
        target_path="result.mp4",
        callback=callback
    )
    ```

<video controls>
    <source src="https://media.roboflow.com/supervision/video-examples/how-to/track-objects/annotate-video-with-tracking-ids.mp4" type="video/mp4">
</video>

### Annotate Video with Traces

Adding traces to the video involves overlaying the historical paths of the detected
objects. This feature, powered by the
[`sv.TraceAnnotator`](/latest/detection/annotators/#supervision.annotators.core.TraceAnnotator),
allows for visualizing the trajectories of objects, helping in understanding the
movement patterns and interactions between objects in the video.

=== "Ultralytics"

    ```{ .py hl_lines="9 26-27" }
    import numpy as np
    import supervision as sv
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        labels = [
            f"#{tracker_id} {results.names[class_id]}"
            for class_id, tracker_id
            in zip(detections.class_id, detections.tracker_id)
        ]

        annotated_frame = box_annotator.annotate(
            frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(
            annotated_frame, detections=detections, labels=labels)
        return trace_annotator.annotate(
            annotated_frame, detections=detections)

    sv.process_video(
        source_path="people-walking.mp4",
        target_path="result.mp4",
        callback=callback
    )
    ```

=== "Inference"

    ```{ .py hl_lines="9 26-27" }
    import numpy as np
    import supervision as sv
    from inference.models.utils import get_roboflow_model

    model = get_roboflow_model(model_id="yolov8n-640", api_key=<ROBOFLOW API KEY>)
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)
        detections = tracker.update_with_detections(detections)

        labels = [
            f"#{tracker_id} {results.names[class_id]}"
            for class_id, tracker_id
            in zip(detections.class_id, detections.tracker_id)
        ]

        annotated_frame = box_annotator.annotate(
            frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(
            annotated_frame, detections=detections, labels=labels)
        return trace_annotator.annotate(
            annotated_frame, detections=detections)

    sv.process_video(
        source_path="people-walking.mp4",
        target_path="result.mp4",
        callback=callback
    )
    ```

<video controls>
    <source src="https://media.roboflow.com/supervision/video-examples/how-to/track-objects/annotate-video-with-traces.mp4" type="video/mp4">
</video>

## Keypoints

Models aren't limited to object detection and segmentation. Keypoint detection allows for detailed analysis of body joints and connections, especially valuable for applications like human pose estimation. This section introduces keypoint tracking. We'll walk through the steps of annotating keypoints, converting them into bounding box detections compatible with `ByteTrack`, and applying detection smoothing for enhanced stability.

To make it easier for you to follow our tutorial, let's download the video we will use as an
example. You can do this using [`supervision[assets]`](/latest/assets/) extension.

```python
from supervision.assets import download_assets, VideoAssets

download_assets(VideoAssets.SKIING)
```

<video controls muted>
    <source src="https://media.roboflow.com/supervision/video-examples/how-to/track-objects/skiing-hd.mp4" type="video/mp4">
</video>

### Keypoint Detection

First, you'll need to obtain predictions from your keypoint detection model. In this tutorial, we are using the YOLOv8 model as an example. However,
Supervision is versatile and compatible with various models. Check this [link](/latest/keypoint/core/) for guidance on how to plug in other models.

We will define a `callback` function, which will process each frame of the video by obtaining model predictions and then annotating the frame based on these predictions.

Let's immediately visualize the results with our [`EdgeAnnotator`](/latest/keypoint/annotators/#supervision.keypoint.annotators.EdgeAnnotator) and [`VertexAnnotator`](https://supervision.roboflow.com/latest/keypoint/annotators/#supervision.keypoint.annotators.VertexAnnotator).

=== "Ultralytics"

    ```{ .py hl_lines="5 10-11" }
    import numpy as np
    import supervision as sv
    from ultralytics import YOLO

    model = YOLO("yolov8m-pose.pt")
    edge_annotator = sv.EdgeAnnotator()
    vertex_annotator = sv.VertexAnnotator()

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        results = model(frame)[0]
        key_points = sv.KeyPoints.from_ultralytics(results)

        annotated_frame = edge_annotator.annotate(
            frame.copy(), key_points=key_points)
        return vertex_annotator.annotate(
            annotated_frame, key_points=key_points)

    sv.process_video(
        source_path="skiing.mp4",
        target_path="result.mp4",
        callback=callback
    )
    ```

=== "Inference"

    ```{ .py hl_lines="5-6 11-12" }
    import numpy as np
    import supervision as sv
    from inference.models.utils import get_roboflow_model

    model = get_roboflow_model(
        model_id="yolov8m-pose-640", api_key=<ROBOFLOW API KEY>)
    edge_annotator = sv.EdgeAnnotator()
    vertex_annotator = sv.VertexAnnotator()

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        results = model.infer(frame)[0]
        key_points = sv.KeyPoints.from_inference(results)

        annotated_frame = edge_annotator.annotate(
            frame.copy(), key_points=key_points)
        return vertex_annotator.annotate(
            annotated_frame, key_points=key_points)

    sv.process_video(
        source_path="skiing.mp4",
        target_path="result.mp4",
        callback=callback
    )
    ```

<video controls>
    <source src="https://media.roboflow.com/supervision/video-examples/how-to/track-objects/track-keypoints-only-keypoints.mp4" type="video/mp4">
</video>

### Convert to Detections

Keypoint tracking is currently supported via the conversion of `KeyPoints` to `Detections`. This is achieved with the [`KeyPoints.as_detections()`](/latest/keypoint/core/#supervision.keypoint.core.KeyPoints.as_detections) function.

Let's convert to detections and visualize the results with our [`BoxAnnotator`](/latest/detection/annotators/#supervision.annotators.core.BoxAnnotator).

!!! tip

    You may use the `selected_keypoint_indices` argument to specify a subset of keypoints to convert. This is useful when some keypoints could be occluded. For example: a person might swing their arm, causing the elbow to be occluded by the torso sometimes.

=== "Ultralytics"

    ```{ .py hl_lines="8 13 19-20" }
    import numpy as np
    import supervision as sv
    from ultralytics import YOLO

    model = YOLO("yolov8m-pose.pt")
    edge_annotator = sv.EdgeAnnotator()
    vertex_annotator = sv.VertexAnnotator()
    box_annotator = sv.BoxAnnotator()

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        results = model(frame)[0]
        key_points = sv.KeyPoints.from_ultralytics(results)
        detections = key_points.as_detections()

        annotated_frame = edge_annotator.annotate(
            frame.copy(), key_points=key_points)
        annotated_frame = vertex_annotator.annotate(
            annotated_frame, key_points=key_points)
        return box_annotator.annotate(
            annotated_frame, detections=detections)

    sv.process_video(
        source_path="skiing.mp4",
        target_path="result.mp4",
        callback=callback
    )
    ```

=== "Inference"

    ```{ .py hl_lines="9 14 20-21" }
    import numpy as np
    import supervision as sv
    from inference.models.utils import get_roboflow_model

    model = get_roboflow_model(
        model_id="yolov8m-pose-640", api_key=<ROBOFLOW API KEY>)
    edge_annotator = sv.EdgeAnnotator()
    vertex_annotator = sv.VertexAnnotator()
    box_annotator = sv.BoxAnnotator()

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        results = model.infer(frame)[0]
        key_points = sv.KeyPoints.from_inference(results)
        detections = key_points.as_detections()

        annotated_frame = edge_annotator.annotate(
            frame.copy(), key_points=key_points)
        annotated_frame = vertex_annotator.annotate(
            annotated_frame, key_points=key_points)
        return box_annotator.annotate(
            annotated_frame, detections=detections)

    sv.process_video(
        source_path="skiing.mp4",
        target_path="result.mp4",
        callback=callback
    )
    ```

<video controls>
    <source src="https://media.roboflow.com/supervision/video-examples/how-to/track-objects/track-keypoints-converted-to-detections.mp4" type="video/mp4">
</video>

### Keypoint Tracking

Now that we have a `Detections` object, we can track it throughout the video. Utilizing Supervision’s [`sv.ByteTrack`](/latest/trackers/#supervision.tracker.byte_tracker.core.ByteTrack) functionality, each detected object is assigned a unique tracker ID, enabling the continuous following of the object's motion path across different frames. We shall visualize the result with `TraceAnnotator`.

=== "Ultralytics"

    ```{ .py hl_lines="10-11 17 25-26" }
    import numpy as np
    import supervision as sv
    from ultralytics import YOLO

    model = YOLO("yolov8m-pose.pt")
    edge_annotator = sv.EdgeAnnotator()
    vertex_annotator = sv.VertexAnnotator()
    box_annotator = sv.BoxAnnotator()

    tracker = sv.ByteTrack()
    trace_annotator = sv.TraceAnnotator()

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        results = model(frame)[0]
        key_points = sv.KeyPoints.from_ultralytics(results)
        detections = key_points.as_detections()
        detections = tracker.update_with_detections(detections)

        annotated_frame = edge_annotator.annotate(
            frame.copy(), key_points=key_points)
        annotated_frame = vertex_annotator.annotate(
            annotated_frame, key_points=key_points)
        annotated_frame = box_annotator.annotate(
            annotated_frame, detections=detections)
        return trace_annotator.annotate(
            annotated_frame, detections=detections)

    sv.process_video(
        source_path="skiing.mp4",
        target_path="result.mp4",
        callback=callback
    )
    ```

=== "Inference"

    ```{ .py hl_lines="11-12 18 26-27" }
    import numpy as np
    import supervision as sv
    from inference.models.utils import get_roboflow_model

    model = get_roboflow_model(
        model_id="yolov8m-pose-640", api_key=<ROBOFLOW API KEY>)
    edge_annotator = sv.EdgeAnnotator()
    vertex_annotator = sv.VertexAnnotator()
    box_annotator = sv.BoxAnnotator()

    tracker = sv.ByteTrack()
    trace_annotator = sv.TraceAnnotator()

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        results = model.infer(frame)[0]
        key_points = sv.KeyPoints.from_inference(results)
        detections = key_points.as_detections()
        detections = tracker.update_with_detections(detections)

        annotated_frame = edge_annotator.annotate(
            frame.copy(), key_points=key_points)
        annotated_frame = vertex_annotator.annotate(
            annotated_frame, key_points=key_points)
        annotated_frame = box_annotator.annotate(
            annotated_frame, detections=detections)
        return trace_annotator.annotate(
            annotated_frame, detections=detections)

    sv.process_video(
        source_path="skiing.mp4",
        target_path="result.mp4",
        callback=callback
    )
    ```

<video controls>
    <source src="https://media.roboflow.com/supervision/video-examples/how-to/track-objects/track-keypoints-with-tracking.mp4" type="video/mp4">
</video>

### Bonus: Smoothing

We could stop here as we have successfully tracked the object detected by the keypoint model. However, we can further enhance the stability of the boxes by applying [`DetectionsSmoother`](/latest/detection/tools/smoother/). This tool helps in stabilizing the boxes by smoothing the bounding box coordinates across frames. It is very simple to use:

=== "Ultralytics"

    ```{ .py hl_lines="11 19" }
    import numpy as np
    import supervision as sv
    from ultralytics import YOLO

    model = YOLO("yolov8m-pose.pt")
    edge_annotator = sv.EdgeAnnotator()
    vertex_annotator = sv.VertexAnnotator()
    box_annotator = sv.BoxAnnotator()

    tracker = sv.ByteTrack()
    smoother = sv.DetectionsSmoother()
    trace_annotator = sv.TraceAnnotator()

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        results = model(frame)[0]
        key_points = sv.KeyPoints.from_ultralytics(results)
        detections = key_points.as_detections()
        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)

        annotated_frame = edge_annotator.annotate(
            frame.copy(), key_points=key_points)
        annotated_frame = vertex_annotator.annotate(
            annotated_frame, key_points=key_points)
        annotated_frame = box_annotator.annotate(
            annotated_frame, detections=detections)
        return trace_annotator.annotate(
            annotated_frame, detections=detections)

    sv.process_video(
        source_path="skiing.mp4",
        target_path="result.mp4",
        callback=callback
    )
    ```

=== "Inference"

    ```{ .py hl_lines="12 20" }
    import numpy as np
    import supervision as sv
    from inference.models.utils import get_roboflow_model

    model = get_roboflow_model(
        model_id="yolov8m-pose-640", api_key=<ROBOFLOW API KEY>)
    edge_annotator = sv.EdgeAnnotator()
    vertex_annotator = sv.VertexAnnotator()
    box_annotator = sv.BoxAnnotator()

    tracker = sv.ByteTrack()
    smoother = sv.DetectionsSmoother()
    trace_annotator = sv.TraceAnnotator()

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        results = model.infer(frame)[0]
        key_points = sv.KeyPoints.from_inference(results)
        detections = key_points.as_detections()
        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)

        annotated_frame = edge_annotator.annotate(
            frame.copy(), key_points=key_points)
        annotated_frame = vertex_annotator.annotate(
            annotated_frame, key_points=key_points)
        annotated_frame = box_annotator.annotate(
            annotated_frame, detections=detections)
        return trace_annotator.annotate(
            annotated_frame, detections=detections)

    sv.process_video(
        source_path="skiing.mp4",
        target_path="result.mp4",
        callback=callback
    )
    ```

<video controls>
    <source src="https://media.roboflow.com/supervision/video-examples/how-to/track-objects/track-keypoints-with-smoothing.mp4" type="video/mp4">
</video>

This structured walkthrough should give a detailed pathway to annotate videos effectively using Supervision’s various functionalities, including object tracking and trace annotations.
