## 👋 hello

This script provides functionality for processing videos using YOLOv8 for object
detection and Supervision for tracking and annotation.

## 💻 install

- clone repository and navigate to example directory

    ```bash
    git clone https://github.com/roboflow/supervision.git
    cd supervision/examples/tracking
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

- `--source_weights_path`: Required. Specifies the path to the YOLO model's weights 
file, which is essential for the object detection process. This file contains the data 
that the model uses to identify objects in the video.
- `--source_video_path`: Required. The path to the source video file to be processed. 
This is the video on which object detection and annotation will be performed.
- `--target_video_path`: Required. The path where the processed video, with annotations
added, will be saved. This is your output video file.
- `--confidence_threshold` (optional): Sets the confidence level at which the model 
identifies objects in the video. Default is `0.3`. A higher threshold makes the model 
more selective, while a lower threshold makes it more inclusive in identifying objects.
- `--iou_threshold` (optional): Specifies the IOU (Intersection Over Union) threshold 
for the model, defaulting to `0.7`. This parameter helps in differentiating between 
distinct objects, especially in crowded scenes.

## ⚙️ run

```bash
python script.py \
--source_weights_path yolov8s.pt \
--source_video_path input.mp4 \
--target_video_path tracking_result.mp4
```

## © license

This demo integrates two main components, each with its own licensing:

YOLOv8: The object detection model used in this demo, YOLOv8, is distributed under the
[AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE). You
can find more details about this license here.

Supervision: The analytics code that powers the tracking and visualization in this demo
is based on the Supervision library, which is licensed under the
[MIT license](https://github.com/roboflow/supervision/blob/develop/LICENSE.md). This
makes the Supervision part of the code fully open source and freely usable in your
projects.
