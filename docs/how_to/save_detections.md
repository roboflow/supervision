---
comments: true
status: new
---

# Save Detections

TODO

## Run Detection

First, you'll need to obtain predictions from your object detection or segmentation
model. You can learn more on this topic in our
[How to Detect and Annotate](/latest/how_to/detect_and_annotate.md) guide.

=== "Inference"

    ```python
    import supervision as sv
    from inference import get_model

    model = get_model(model_id="yolov8n-640")
    frames_generator = sv.get_video_frames_generator(<SOURCE_VIDEO_PATH>)

    for frame in frames_generator:

        results = model.infer(image)[0]
        detections = sv.Detections.from_inference(results)
    ```

=== "Ultralytics"

    ```python
    import supervision as sv
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    frames_generator = sv.get_video_frames_generator(<SOURCE_VIDEO_PATH>)

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
    frames_generator = sv.get_video_frames_generator(<SOURCE_VIDEO_PATH>)

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
    ```

## Save Detections as CSV

TODO

=== "Inference"

    ```{ .py hl_lines="7 12" }
    import supervision as sv
    from inference import get_model

    model = get_model(model_id="yolov8n-640")
    frames_generator = sv.get_video_frames_generator(<SOURCE_VIDEO_PATH>)

    with sv.CSVSink(<TARGET_CSV_PATH>) as sink:
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
    frames_generator = sv.get_video_frames_generator(<SOURCE_VIDEO_PATH>)

    with sv.CSVSink(<TARGET_CSV_PATH>) as sink:
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
    frames_generator = sv.get_video_frames_generator(<SOURCE_VIDEO_PATH>)

    with sv.CSVSink(<TARGET_CSV_PATH>) as sink:
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

TODO

| x_min   | y_min    | x_max   | y_max    | class_id | confidence | tracker_id | class_name |
|---------|----------|---------|----------|----------|------------|------------|------------|
| 2941.14 | 1269.31  | 3220.77 | 1500.67  | 2        | 0.8517     |            | car        |
| 944.889 | 899.641  | 1235.42 | 1308.80  | 7        | 0.6752     |            | truck      |
| 1439.78 | 1077.79  | 1621.27 | 1231.40  | 2        | 0.6450     |            | car        |

## Custom Fields

TODO

=== "Inference"

    ```{ .py hl_lines="8 12" }
    import supervision as sv
    from inference import get_model

    model = get_model(model_id="yolov8n-640")
    frames_generator = sv.get_video_frames_generator(<SOURCE_VIDEO_PATH>)

    with sv.CSVSink(<TARGET_CSV_PATH>) as sink:
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
    frames_generator = sv.get_video_frames_generator(<SOURCE_VIDEO_PATH>)

    with sv.CSVSink(<TARGET_CSV_PATH>) as sink:
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
    frames_generator = sv.get_video_frames_generator(<SOURCE_VIDEO_PATH>)

    with sv.CSVSink(<TARGET_CSV_PATH>) as sink:
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

TODO

| x_min   | y_min    | x_max   | y_max    | class_id | confidence | tracker_id | class_name | frame_index |
|---------|----------|---------|----------|----------|------------|------------|------------|-------------|
| 2941.14 | 1269.31  | 3220.77 | 1500.67  | 2        | 0.8517     |            | car        | 0           |
| 944.889 | 899.641  | 1235.42 | 1308.80  | 7        | 0.6752     |            | truck      | 0           |
| 1439.78 | 1077.79  | 1621.27 | 1231.40  | 2        | 0.6450     |            | car        | 0           |

## Save Detections as JSON

TODO

=== "Inference"

    ```{ .py hl_lines="7" }
    import supervision as sv
    from inference import get_model

    model = get_model(model_id="yolov8n-640")
    frames_generator = sv.get_video_frames_generator(<SOURCE_VIDEO_PATH>)

    with sv.JSONSink(<TARGET_CSV_PATH>) as sink:
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
    frames_generator = sv.get_video_frames_generator(<SOURCE_VIDEO_PATH>)

    with sv.JSONSink(<TARGET_CSV_PATH>) as sink:
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
    frames_generator = sv.get_video_frames_generator(<SOURCE_VIDEO_PATH>)

    with sv.JSONSink(<TARGET_CSV_PATH>) as sink:
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
