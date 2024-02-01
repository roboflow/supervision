# time in zone

## 👋 hello

This is a demo for real-time object detection and tracking in predefined polygonal 
zones, using a webcam as the video source. It features functionalities for loading zone 
configurations, processing webcam frames, tracking objects, and annotating these frames 
with detection information, including the time duration each object remains within a 
zone.

## 💻 install

- clone repository and navigate to example directory

  ```bash
  git clone https://github.com/roboflow/supervision.git
  cd supervision/examples/time_in_zone
  ```

- setup python environment and activate it [optional]

  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

- install required dependencies

  ```bash
  pip install -r requirements.txt
  ```
  
## 🛠️ script arguments

- ultralytics

  - `--source_weights_path` (optional): The path to the YOLO model's weights file.
    Defaults to `"yolov8m.pt"` if not specified.
  - `--device` (optional): This argument allows the user to specify the computing device
    for processing the video and object detection tasks. The options are `"cuda"`, 
    `"cpu"`, and `"mps"`. The default setting is `"cpu"`.

  - `--camera_index` (optional): An integer representing the index of the webcam to be 
    used for capturing video. If not specified, it defaults to `0`, which usually 
    corresponds to the primary webcam on a device.
  - `--zone_configuration_path`: Specifies the path to the JSON file containing zone
    configurations. This file defines the polygonal areas in the video where objects will
    be counted.
  - `--confidence_threshold` (optional): Sets the confidence threshold for the YOLO model
    to filter detections. Default is `0.3`.
  - `--iou_threshold` (optional): Specifies the IOU (Intersection Over Union) threshold
    for the model. Default is `0.7`.

- inference

  - `--model_id` (optional): Designates the Roboflow model ID to be used. The default
    value is `"yolov8m-640"`.

  - `--camera_index` (optional): An integer representing the index of the webcam to be 
    used for capturing video. If not specified, it defaults to `0`, which usually 
    corresponds to the primary webcam on a device.
  - `--zone_configuration_path`: Specifies the path to the JSON file containing zone
    configurations. This file defines the polygonal areas in the video where objects will
    be counted.
  - `--confidence_threshold` (optional): Sets the confidence threshold for the YOLO model
    to filter detections. Default is `0.3`.
  - `--iou_threshold` (optional): Specifies the IOU (Intersection Over Union) threshold
    for the model. Default is `0.7`.

## ⚙️ run

- ultralytics

    ```bash
    python ultralytics_webcam.py \
      --camera_index 0 \
      --zone_configuration_path data/zones.json \
      --source_weights_path yolov8m.pt \
      --device cpu \
      --confidence_threshold 0.3 \
      --iou_threshold 0.5
    ```
  
- inference
- 
    ```bash
    python inference_webcam.py \
      --camera_index 0 \
      --zone_configuration_path data/zones.json \
      --model_id yolov8m-640 \
      --confidence_threshold 0.3 \
      --iou_threshold 0.5
    ```

## © license

This demo integrates two main components, each with its own licensing:

- ultralytics: The object detection model used in this demo, YOLOv8, is distributed
  under the [AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
  You can find more details about this license here.

- supervision: The analytics code that powers the zone-based analysis in this demo is
  based on the Supervision library, which is licensed under the
  [MIT license](https://github.com/roboflow/supervision/blob/develop/LICENSE.md). This
  makes the Supervision part of the code fully open source and freely usable in your
  projects.
