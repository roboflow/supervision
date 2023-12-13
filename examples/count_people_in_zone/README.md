## 👋 hello

This demo is a video analysis tool that counts and highlights objects in specific zones
of a video. Each zone and the objects within it are marked in different colors, making
it easy to see and count the objects in each area. The tool can save this enhanced
video or display it live on the screen.

## 💻 install

- clone repository and navigate to example directory

    ```bash
    git clone https://github.com/roboflow/supervision.git
    cd supervision/examples/count_people_in_zone
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

- `--zone_configuration_path`: Specifies the path to the JSON file containing zone 
configurations. This file defines the polygonal areas in the video where objects will 
be counted.
- `--source_weights_path` (optional): The path to the YOLO model's weights file. 
Defaults to `"yolov8x.pt"` if not specified.
- `--source_video_path`: The path to the source video file that will be analyzed.
- `--target_video_path` (optional): The path to save the output video with annotations. 
If not provided, the processed video will be displayed in real-time.
- `--confidence_threshold` (optional): Sets the confidence threshold for the YOLO model 
to filter detections. Default is `0.3`.
- `--iou_threshold` (optional): Specifies the IOU (Intersection Over Union) threshold 
for the model. Default is `0.7`.

## ⚙️ run example

```bash
python script.py \
--zone_configuration_path data/multi-zone-config.json \
--source_video_path data/market-square.mp4 \
--confidence_threshold 0.3 \
--iou_threshold 0.5
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
