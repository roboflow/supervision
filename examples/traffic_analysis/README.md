# traffic analysis

## 👋 hello

This script performs traffic flow analysis using YOLOv8, an object-detection method and
ByteTrack, a simple yet effective online multi-object tracking method. It uses the
supervision package for multiple tasks such as tracking, annotations, etc.

https://github.com/roboflow/supervision/assets/26109316/c9436828-9fbf-4c25-ae8c-60e9c81b3900

## 💻 install

- clone repository and navigate to example directory

    ```bash
    git clone https://github.com/roboflow/supervision.git
    cd supervision/examples/traffic_analysis
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

- download `traffic_analysis.pt` and `traffic_analysis.mov` files

    ```bash
    ./setup.sh
    ```

## 🛠️ script arguments

- `--source_weights_path`: Required. Specifies the path to the weights file for the
YOLO model. This file contains the trained model data necessary for object detection.
- `--source_video_path`: Required. The path to the source video file that will be
analyzed. This is the input video on which traffic flow analysis will be performed.
- `--target_video_path` (optional): The path to save the output video with annotations.
If not specified, the processed video will be displayed in real-time without being
saved.
- `--confidence_threshold` (optional): Sets the confidence threshold for the YOLO model
to filter detections. Default is `0.3`. This determines how confident the model should
be to recognize an object in the video.
- `--iou_threshold` (optional): Specifies the IOU (Intersection Over Union) threshold
for the model. Default is 0.7. This value is used to manage object detection accuracy,
particularly in distinguishing between different objects.

## ⚙️ run

```bash
python script.py \
--source_weights_path data/traffic_analysis.pt \
--source_video_path data/traffic_analysis.mov \
--confidence_threshold 0.3 \
--iou_threshold 0.5 \
--target_video_path data/traffic_analysis_result.mov
```

## © license

This demo integrates two main components, each with its own licensing:

YOLOv8: The object detection model used in this demo, YOLOv8, is distributed under the
[AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE). You
can find more details about this license here.

Supervision: The analytics code that powers the zone-based analysis in this demo is
based on the Supervision library, which is licensed under the
[MIT license](https://github.com/roboflow/supervision/blob/develop/LICENSE.md). This
makes the Supervision part of the code fully open source and freely usable in your
projects.
