---
comments: true
description: Save object detection results to CSV or JSON with supervision's CSVSink and JSONSink — export predictions for analysis and downstream pipelines.
authors:
  - name: SkalskiP (Piotr Skalski)
    role: Computer Vision Engineer, Roboflow
    github: https://github.com/SkalskiP
date_modified: 2026-04-22
---

# Save Detections

Supervision enables an easy way to save detections in .CSV and .JSON files for offline
processing. This guide demonstrates how to perform video inference using the
[Inference](https://github.com/roboflow/inference),
[Ultralytics](https://github.com/ultralytics/ultralytics) or
[Transformers](https://github.com/huggingface/transformers) packages and save their results with
[`sv.CSVSink`](https://supervision.roboflow.com/latest/detection/tools/save_detections/#supervision.detection.tools.csv_sink.CSVSink) and
[`sv.JSONSink`](https://supervision.roboflow.com/latest/detection/tools/save_detections/#supervision.detection.tools.json_sink.JSONSink).

## Run Detection

First, you'll need to obtain predictions from your object detection or segmentation
model. You can learn more on this topic in our
[How to Detect and Annotate](https://supervision.roboflow.com/latest/how_to/detect_and_annotate/) guide.

To generate predictions for saving, initialize your model and iterate over video frames using `sv.get_video_frames_generator`. Each frame is passed to the model, and the raw output is converted into a `sv.Detections` object. This detection loop forms the foundation for both CSV and JSON export workflows shown below.

=== "Inference"

    ```python
    import supervision as sv
    from inference import get_model

    model = get_model(model_id="yolov8n-640")
    frames_generator = sv.get_video_frames_generator("<SOURCE_VIDEO_PATH>")

    for frame in frames_generator:
        results = model.infer(image)[0]
        detections = sv.Detections.from_inference(results)
    ```

=== "Ultralytics"

    ```python
    import supervision as sv
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    frames_generator = sv.get_video_frames_generator("<SOURCE_VIDEO_PATH>")

    for frame in frames_generator:
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
    ```

=== "Transformers"

    ```python
    import torch
    import supervision as sv
    from transformers import DetrImageProcessor, DetrForObjectDetection

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    frames_generator = sv.get_video_frames_generator("<SOURCE_VIDEO_PATH>")

    for frame in frames_generator:
        frame = sv.cv2_to_pillow(frame)
        inputs = processor(images=frame, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        width, height = frame.size
        target_size = torch.tensor([[height, width]])
        results = processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_size
        )[0]
        detections = sv.Detections.from_transformers(results)
    ```

## Save Detections as CSV

To save detections to a `.CSV` file, open our
[`sv.CSVSink`](https://supervision.roboflow.com/latest/detection/tools/save_detections/#supervision.detection.tools.csv_sink.CSVSink)
and then pass the
[`sv.Detections`](https://supervision.roboflow.com/latest/detection/core/#supervision.detection.core.Detections)
object resulting from the inference to it. Its fields are parsed and saved on disk.

=== "Inference"

    ```{ .py hl_lines="7 12" }
    import supervision as sv
    from inference import get_model

    model = get_model(model_id="yolov8n-640")
    frames_generator = sv.get_video_frames_generator("<SOURCE_VIDEO_PATH>")

    with sv.CSVSink("<TARGET_CSV_PATH>") as sink:
        for frame in frames_generator:

            results = model.infer(image)[0]
            detections = sv.Detections.from_inference(results)
            sink.append(detections, {})
    ```

=== "Ultralytics"

    ```{ .py hl_lines="7 12" }
    import supervision as sv
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    frames_generator = sv.get_video_frames_generator("<SOURCE_VIDEO_PATH>")

    with sv.CSVSink("<TARGET_CSV_PATH>") as sink:
        for frame in frames_generator:

            results = model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)
            sink.append(detections, {})
    ```

=== "Transformers"

    ```{ .py hl_lines="9 23" }
    import torch
    import supervision as sv
    from transformers import DetrImageProcessor, DetrForObjectDetection

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    frames_generator = sv.get_video_frames_generator("<SOURCE_VIDEO_PATH>")

    with sv.CSVSink("<TARGET_CSV_PATH>") as sink:
        for frame in frames_generator:

            frame = sv.cv2_to_pillow(frame)
            inputs = processor(images=frame, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            width, height = frame.size
            target_size = torch.tensor([[height, width]])
            results = processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_size)[0]
            detections = sv.Detections.from_transformers(results)
            sink.append(detections, {})
    ```

| x_min   | y_min   | x_max   | y_max   | class_id | confidence | tracker_id | class_name |
| ------- | ------- | ------- | ------- | -------- | ---------- | ---------- | ---------- |
| 2941.14 | 1269.31 | 3220.77 | 1500.67 | 2        | 0.8517     |            | car        |
| 944.889 | 899.641 | 1235.42 | 1308.80 | 7        | 0.6752     |            | truck      |
| 1439.78 | 1077.79 | 1621.27 | 1231.40 | 2        | 0.6450     |            | car        |

## Custom Fields

Besides regular fields in
[`sv.Detections`](https://supervision.roboflow.com/latest/detection/core/#supervision.detection.core.Detections),
[`sv.CSVSink`](https://supervision.roboflow.com/latest/detection/tools/save_detections/#supervision.detection.tools.csv_sink.CSVSink)
also allows you to add custom information to each row, which can be passed via the
`custom_data` dictionary. Let's utilize this feature to save information about the
frame index from which the detections originate.

=== "Inference"

    ```{ .py hl_lines="8 12" }
    import supervision as sv
    from inference import get_model

    model = get_model(model_id="yolov8n-640")
    frames_generator = sv.get_video_frames_generator("<SOURCE_VIDEO_PATH>")

    with sv.CSVSink("<TARGET_CSV_PATH>") as sink:
        for frame_index, frame in enumerate(frames_generator):

            results = model.infer(image)[0]
            detections = sv.Detections.from_inference(results)
            sink.append(detections, {"frame_index": frame_index})
    ```

=== "Ultralytics"

    ```{ .py hl_lines="8 12" }
    import supervision as sv
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    frames_generator = sv.get_video_frames_generator("<SOURCE_VIDEO_PATH>")

    with sv.CSVSink("<TARGET_CSV_PATH>") as sink:
        for frame_index, frame in enumerate(frames_generator):

            results = model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)
            sink.append(detections, {"frame_index": frame_index})
    ```

=== "Transformers"

    ```{ .py hl_lines="10 23" }
    import torch
    import supervision as sv
    from transformers import DetrImageProcessor, DetrForObjectDetection

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    frames_generator = sv.get_video_frames_generator("<SOURCE_VIDEO_PATH>")

    with sv.CSVSink("<TARGET_CSV_PATH>") as sink:
        for frame_index, frame in enumerate(frames_generator):

            frame = sv.cv2_to_pillow(frame)
            inputs = processor(images=frame, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            width, height = frame.size
            target_size = torch.tensor([[height, width]])
            results = processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_size)[0]
            detections = sv.Detections.from_transformers(results)
            sink.append(detections, {"frame_index": frame_index})
    ```

| x_min   | y_min   | x_max   | y_max   | class_id | confidence | tracker_id | class_name | frame_index |
| ------- | ------- | ------- | ------- | -------- | ---------- | ---------- | ---------- | ----------- |
| 2941.14 | 1269.31 | 3220.77 | 1500.67 | 2        | 0.8517     |            | car        | 0           |
| 944.889 | 899.641 | 1235.42 | 1308.80 | 7        | 0.6752     |            | truck      | 0           |
| 1439.78 | 1077.79 | 1621.27 | 1231.40 | 2        | 0.6450     |            | car        | 0           |

## Save Detections as JSON

If you prefer to save the result in a `.JSON` file instead of a `.CSV` file, all you
need to do is replace
[`sv.CSVSink`](https://supervision.roboflow.com/latest/detection/tools/save_detections/#supervision.detection.tools.csv_sink.CSVSink)
with
[`sv.JSONSink`](https://supervision.roboflow.com/latest/detection/tools/save_detections/#supervision.detection.tools.json_sink.JSONSink).

=== "Inference"

    ```{ .py hl_lines="7" }
    import supervision as sv
    from inference import get_model

    model = get_model(model_id="yolov8n-640")
    frames_generator = sv.get_video_frames_generator("<SOURCE_VIDEO_PATH>")

    with sv.JSONSink("<TARGET_JSON_PATH>") as sink:
        for frame_index, frame in enumerate(frames_generator):

            results = model.infer(image)[0]
            detections = sv.Detections.from_inference(results)
            sink.append(detections, {"frame_index": frame_index})
    ```

=== "Ultralytics"

    ```{ .py hl_lines="7" }
    import supervision as sv
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    frames_generator = sv.get_video_frames_generator("<SOURCE_VIDEO_PATH>")

    with sv.JSONSink("<TARGET_JSON_PATH>") as sink:
        for frame_index, frame in enumerate(frames_generator):

            results = model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)
            sink.append(detections, {"frame_index": frame_index})
    ```

=== "Transformers"

    ```{ .py hl_lines="9" }
    import torch
    import supervision as sv
    from transformers import DetrImageProcessor, DetrForObjectDetection

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    frames_generator = sv.get_video_frames_generator("<SOURCE_VIDEO_PATH>")

    with sv.JSONSink("<TARGET_JSON_PATH>") as sink:
        for frame_index, frame in enumerate(frames_generator):

            frame = sv.cv2_to_pillow(frame)
            inputs = processor(images=frame, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            width, height = frame.size
            target_size = torch.tensor([[height, width]])
            results = processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_size)[0]
            detections = sv.Detections.from_transformers(results)
            sink.append(detections, {"frame_index": frame_index})
    ```

## Frequently Asked Questions

### How do I save detections to CSV with supervision?

Open `sv.CSVSink("output.csv")` as a context manager and call `sink.append(detections)` for each frame. The CSV includes box coordinates, confidence, class ID, tracker ID, and any fields stored in `detections.data`.

### Can I save detections to JSON instead?

Yes. Open `sv.JSONSink("output.json")` as a context manager and call `sink.append(detections)` for each frame. The file is written as a JSON array when the context exits.

### Can I add custom fields to the saved output?

Yes. Pass a dict as the second argument: `sink.append(detections, {"frame_index": 5})` — the keys become extra columns in the CSV or extra fields in the JSON.

### Can I save only specific classes or confidence levels?

Filter the `Detections` object before saving: `sink.append(detections[detections.confidence > 0.7])`.

## Author

- [Piotr Skalski](https://github.com/SkalskiP) — Computer Vision Engineer, Roboflow
