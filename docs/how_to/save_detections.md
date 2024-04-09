---
comments: true
status: new
---

# Save Detections

TODO

## Run Detection

=== "Inference"

    ```python
    import cv2
    from inference import get_model

    model = get_model(model_id="yolov8n-640")
    image = cv2.imread(<SOURCE_IMAGE_APTH>)
    results = model.infer(image)[0]
    ```

=== "Ultralytics"

    ```python
    import cv2
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    image = cv2.imread(<SOURCE_IMAGE_APTH>)
    results = model(image)[0]
    ```

=== "Transformers"

    ```python
    import torch
    from PIL import Image
    from transformers import DetrImageProcessor, DetrForObjectDetection

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    image = Image.open(<SOURCE_IMAGE_APTH>)
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    width, height = image.size
    target_size = torch.tensor([[height, width]])
    results = processor.post_process_object_detection(
        outputs=outputs, target_sizes=target_size)[0]
    ```

## Save Detections as CSV

TODO

=== "Inference"

    ```python
    import supervision as sv
    from inference import get_model

    model = get_model(model_id="yolov8n-640")

    with sv.CSVSink(<TARGET_CSV_PATH>) as sink:
        for frame in sv.get_video_frames_generator(<SOURCE_VIDEO_PATH>):

            results = model.infer(image)[0]
            detections = sv.Detections.from_inference(results)
            sink.append(detections, {})
    ```

=== "Ultralytics"

    ```python
    import supervision as sv
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")

    with sv.CSVSink(<TARGET_CSV_PATH>) as sink:
        for frame in sv.get_video_frames_generator(<SOURCE_VIDEO_PATH>):

            results = model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)
            sink.append(detections, {})
    ```

=== "Transformers"

    ```python
    import torch
    from PIL import Image
    from transformers import DetrImageProcessor, DetrForObjectDetection

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    image = Image.open(<SOURCE_IMAGE_APTH>)
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    width, height = image.size
    target_size = torch.tensor([[height, width]])
    results = processor.post_process_object_detection(
        outputs=outputs, target_sizes=target_size)[0]
    ```

## Custom Fields

TODO

## Save Detections as JSON

TODO

## Process Video and Save Detections

TODO
