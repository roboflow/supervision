# heatmap and tracking

## 👋 hello

This script performs heatmap and tracking analysis using an object-detection model and ByteTrack, a simple yet effective online multi-object tracking method. It uses the supervision package for multiple tasks such as drawing heatmap annotations, tracking objects, etc. [RF-DETR](https://github.com/roboflow/rf-detr) (`rfdetr_example.py`) is the recommended model — its `predict` method returns a `Detections` object directly, no conversion step needed, and it needs no separate weights file. YOLOv8 via Ultralytics (`script.py`) is also supported, and requires supplying your own model weights file.

## 💻 install

- clone repository and navigate to example directory

  ```bash
  git clone --depth 1 -b develop https://github.com/roboflow/supervision.git
  cd supervision/examples/heatmap_and_track
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

- rfdetr (`rfdetr_example.py`)

  - `--source_video_path` (optional): The path to the source video file that will be analyzed. This is the input video on which crowd analysis will be performed. If not specified default is `people-walking.mp4` from supervision assets
  - `--target_video_path` (optional): The path to save the output.mp4 video with annotations.
  - `--device` (optional): Computation device (`cpu`, `mps` or `cuda`). Default is `cpu`.
  - `--confidence_threshold` (optional): Sets the confidence threshold for the RF-DETR model to filter detections. Default is `0.35`. This determines how confident the model should be to recognize an object in the video.
  - `--iou_threshold` (optional): Specifies the IOU (Intersection Over Union) threshold used for non-max suppression. Default is 0.5. This value is used to manage object detection accuracy, particularly in distinguishing between different objects.
  - `--heatmap_alpha` (optional): Opacity of the overlay mask, between 0 and 1.
  - `--radius` (optional): Radius of the heat circle.
  - `--track_activation_threshold` (optional): Detection confidence threshold for track activation.
  - `--track_seconds` (optional): Number of seconds to buffer when a track is lost.
  - `--minimum_matching_threshold` (optional): Threshold for matching tracks with detections.

- ultralytics (`script.py`)

  - `--source_weights_path`: Required. Specifies the path to the weights file for the YOLO model. This file contains the trained model data necessary for object detection.
  - `--source_video_path` (optional): The path to the source video file that will be analyzed. This is the input video on which crowd analysis will be performed. If not specified default is `people-walking.mp4` from supervision assets
  - `--target_video_path` (optional): The path to save the output.mp4 video with annotations.
  - `--confidence_threshold` (optional): Sets the confidence threshold for the YOLO model to filter detections. Default is `0.3`. This determines how confident the model should be to recognize an object in the video.
  - `--iou_threshold` (optional): Specifies the IOU (Intersection Over Union) threshold for the model. Default is 0.7. This value is used to manage object detection accuracy, particularly in distinguishing between different objects.
  - `--heatmap_alpha` (optional): Opacity of the overlay mask, between 0 and 1.
  - `--radius` (optional): Radius of the heat circle.
  - `--track_activation_threshold` (optional): Detection confidence threshold for track activation.
  - `--track_seconds` (optional): Number of seconds to buffer when a track is lost.
  - `--minimum_matching_threshold` (optional): Threshold for matching tracks with detections.

## ⚙️ run

- rfdetr

  ```bash
  python rfdetr_example.py \
      --source_video_path  input_video.mp4 \
      --confidence_threshold 0.3 \
      --iou_threshold 0.5 \
      --target_video_path  output_video.mp4
  ```

- ultralytics

  ```bash
  python script.py \
      --source_weights_path weight.pt \
      --source_video_path  input_video.mp4 \
      --confidence_threshold 0.3 \
      --iou_threshold 0.5 \
      --target_video_path  output_video.mp4
  ```

## © license

This demo integrates multiple components, each with its own licensing:

- rfdetr: The object detection model used by the recommended `rfdetr_example.py` variant of this demo, [RF-DETR](https://github.com/roboflow/rf-detr), is distributed under the permissive [Apache-2.0 license](https://github.com/roboflow/rf-detr/blob/main/LICENSE).

- ultralytics: The object detection model used by the `script.py` variant of this demo, YOLOv8, is distributed under the [AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE). You can find more details about this license here.

- supervision: The analytics code that powers the zone-based analysis in this demo is based on the Supervision library, which is licensed under the [MIT license](https://github.com/roboflow/supervision/blob/develop/LICENSE.md). This makes the Supervision part of the code fully open source and freely usable in your projects.
