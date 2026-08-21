# tracking

## 👋 hello

This script provides functionality for processing videos using an object-detection model and Supervision for tracking and annotation. [RF-DETR](https://github.com/roboflow/rf-detr) is the recommended model — its `predict` method returns a `Detections` object directly, no conversion step needed. YOLOv8 (via Ultralytics) and Roboflow Inference are also supported as alternatives.

## 💻 install

- clone repository and navigate to example directory

  ```bash
  git clone --depth 1 -b develop https://github.com/roboflow/supervision.git
  cd supervision/examples/tracking
  ```

- setup python environment and activate it [optional]

  ```bash
  uv venv
  source .venv/bin/activate
  ```

- install required dependencies

  ```bash
  uv pip install -r requirements.txt
  ```

## 🛠️ script arguments

- rfdetr

  - `--source_video_path`: Required. The path to the source video file to be processed. This is the video on which object detection and annotation will be performed.

  - `--target_video_path`: Required. The path where the processed video, with annotations added, will be saved. This is your output video file.

  - `--device` (optional): Computation device (`cpu`, `mps` or `cuda`). Default is `cpu`.

  - `--confidence_threshold` (optional): Sets the confidence level at which the model identifies objects in the video. Default is `0.3`. A higher threshold makes the model more selective, while a lower threshold makes it more inclusive in identifying objects.

  - `--iou_threshold` (optional): Specifies the IOU (Intersection Over Union) threshold used for non-max suppression, defaulting to `0.7`. This parameter helps in differentiating between distinct objects, especially in crowded scenes.

- ultralytics

  - `--source_weights_path`: Required. Specifies the path to the YOLO model's weights file, which is essential for the object detection process. This file contains the data that the model uses to identify objects in the video.

  - `--source_video_path`: Required. The path to the source video file to be processed. This is the video on which object detection and annotation will be performed.

  - `--target_video_path`: Required. The path where the processed video, with annotations added, will be saved. This is your output video file.

  - `--confidence_threshold` (optional): Sets the confidence level at which the model identifies objects in the video. Default is `0.3`. A higher threshold makes the model more selective, while a lower threshold makes it more inclusive in identifying objects.

  - `--iou_threshold` (optional): Specifies the IOU (Intersection Over Union) threshold for the model, defaulting to `0.7`. This parameter helps in differentiating between distinct objects, especially in crowded scenes.

- inference

  - `--roboflow_api_key` (optional): The API key for Roboflow services. If not provided directly, the script tries to fetch it from the `ROBOFLOW_API_KEY` environment variable. Follow [this guide](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key) to acquire your `API KEY`.

  - `--model_id` (optional): Designates the Roboflow model ID to be used. The default value is `"rfdetr-small"`.

  - `--source_video_path`: Required. The path to the source video file to be processed. This is the video on which object detection and annotation will be performed.

  - `--target_video_path`: Required. The path where the processed video, with annotations added, will be saved. This is your output video file.

  - `--confidence_threshold` (optional): Sets the confidence level at which the model identifies objects in the video. Default is `0.3`. A higher threshold makes the model more selective, while a lower threshold makes it more inclusive in identifying objects.

  - `--iou_threshold` (optional): Specifies the IOU (Intersection Over Union) threshold for the model, defaulting to `0.7`. This parameter helps in differentiating between distinct objects, especially in crowded scenes.

## ⚙️ run

- rfdetr

  ```bash
  python rfdetr_example.py \
      --source_video_path input.mp4 \
      --target_video_path tracking_result.mp4
  ```

- inference

  ```bash
  python inference_example.py \
      --roboflow_api_key "ROBOFLOW_API_KEY" \
      --source_video_path input.mp4 \
      --target_video_path tracking_result.mp4
  ```

- ultralytics

  ```bash
  python ultralytics_example.py \
      --source_weights_path yolov8s.pt \
      --source_video_path input.mp4 \
      --target_video_path tracking_result.mp4
  ```

## © license

This demo integrates multiple components, each with its own licensing:

- rfdetr: The object detection model used by the recommended variant of this demo, [RF-DETR](https://github.com/roboflow/rf-detr), is distributed under the permissive [Apache-2.0 license](https://github.com/roboflow/rf-detr/blob/main/LICENSE).

- ultralytics: The object detection model used by the `ultralytics` variant of this demo, YOLOv8, is distributed under the [AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE). You can find more details about this license here.

- supervision: The analytics code that powers the zone-based analysis in this demo is based on the Supervision library, which is licensed under the [MIT license](https://github.com/roboflow/supervision/blob/develop/LICENSE.md). This makes the Supervision part of the code fully open source and freely usable in your projects.
