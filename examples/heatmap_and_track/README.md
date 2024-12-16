# heatmap and tracking

## 👋 hello

This script performs heatmap and tracking analysis using YOLOv8, an object-detection method and
ByteTrack, a simple yet effective online multi-object tracking method. It uses the
supervision package for multiple tasks such as drawing heatmap annotations, tracking objects, etc.

## 💻 install

- clone repository and navigate to example directory

    ```bash
    git clone --depth 1 -b develop https://github.com/roboflow/supervision.git
    cd supervision/examples/heatmap_and_track
    ```

- setup python environment and activate it \[optional\]

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

- install required dependencies

    ```bash
    pip install -r requirements.txt
    ```

## 🛠️ script arguments

- `--source_weights_path`: Required. Specifies the path to the weights file for the
    YOLO model. This file contains the trained model data necessary for object detection.
- `--source_video_path` (optional): The path to the source video file that will be
    analyzed. This is the input video on which crowd analysis will be performed.
    If not specified default is `people-walking.mp4` from supervision assets
- `--target_video_path` (optional): The path to save the output.mp4 video with annotations.
- `--confidence_threshold` (optional): Sets the confidence threshold for the YOLO model
    to filter detections. Default is `0.3`. This determines how confident the model should
    be to recognize an object in the video.
- `--iou_threshold` (optional): Specifies the IOU (Intersection Over Union) threshold
    for the model. Default is 0.7. This value is used to manage object detection accuracy,
    particularly in distinguishing between different objects.
- `--heatmap_alpha` (optional): Opacity of the overlay mask, between 0 and 1.
- `--radius` (optional): Radius of the heat circle.
- `--track_threshold` (optional): Detection confidence threshold for track activation.
- `--track_seconds` (optional): Number of seconds to buffer when a track is lost.
- `--match_threshold` (optional): Threshold for matching tracks with detections.

## ⚙️ run

```bash
python script.py \
    --source_weights_path weight.pt \
    --source_video_path  input_video.mp4 \
    --confidence_threshold 0.3 \
    --iou_threshold 0.5 \
    --target_video_path  output_video.mp4
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
