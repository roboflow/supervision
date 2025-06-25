# count people in zone

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-detect-and-count-objects-in-polygon-zone.ipynb)
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=l_kf9CfZ_8M)

## 👋 hello

This demo is a video analysis tool that counts and highlights objects in specific zones
of a video. Each zone and the objects within it are marked in different colors, making
it easy to see and count the objects in each area. The tool can save this enhanced
video or display it live on the screen.

https://github.com/roboflow/supervision/assets/26109316/f84db7b5-79e2-4142-a1da-64daa43ce667

## 💻 install

- clone repository and navigate to example directory

    ```bash
    git clone --depth 1 -b develop https://github.com/roboflow/supervision.git
    cd supervision/examples/count_people_in_zone
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

- download `traffic_analysis.pt` and `traffic_analysis.mov` files

    ```bash
    ./setup.sh
    ```

## 🛠️ script arguments

- ultralytics

    - `--source_weights_path` (optional): The path to the YOLO model's weights file.
        Defaults to `"yolov8x.pt"` if not specified.

    - `--zone_configuration_path`: Specifies the path to the JSON file containing zone
        configurations. This file defines the polygonal areas in the video where objects will
        be counted.

    - `--source_video_path`: The path to the source video file that will be analyzed.

    - `--target_video_path` (optional): The path to save the output video with annotations.
        If not provided, the processed video will be displayed in real-time.

    - `--confidence_threshold` (optional): Sets the confidence threshold for the YOLO model
        to filter detections. Default is `0.3`.

    - `--iou_threshold` (optional): Specifies the IOU (Intersection Over Union) threshold
        for the model. Default is `0.7`.

- inference

    - `--roboflow_api_key` (optional): The API key for Roboflow services. If not provided
        directly, the script tries to fetch it from the `ROBOFLOW_API_KEY` environment
        variable. Follow [this guide](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key)
        to acquire your `API KEY`.

    - `--model_id` (optional): Designates the Roboflow model ID to be used. The default
        value is `"yolov8x-1280"`.

    - `--zone_configuration_path`: Specifies the path to the JSON file containing zone
        configurations. This file defines the polygonal areas in the video where objects will
        be counted.

    - `--source_video_path`: The path to the source video file that will be analyzed.

    - `--target_video_path` (optional): The path to save the output video with annotations.
        If not provided, the processed video will be displayed in real-time.

    - `--confidence_threshold` (optional): Sets the confidence threshold for the YOLO model
        to filter detections. Default is `0.3`.

    - `--iou_threshold` (optional): Specifies the IOU (Intersection Over Union) threshold
        for the model. Default is `0.7`.

## 📌 zone configuration

- `horizontal-zone-config.json`: Defines zones divided horizontally across the frame.
- `multi-zone-config.json`: Configures multiple zones with custom shapes and positions.
- `quarters-zone-config.json`: Splits the frame into four equal quarters.
- `vertical-zone-config.json`: Divides the frame into vertical zones of equal width.

## ⚙️ run example

- ultralytics

    ```bash
    python ultralytics_example.py \
        --zone_configuration_path data/multi-zone-config.json \
        --source_video_path data/market-square.mp4 \
        --confidence_threshold 0.3 \
        --iou_threshold 0.5
    ```

- inference

    ```bash
    python inference_example.py \
        --roboflow_api_key <ROBOFLOW API KEY> \
        --zone_configuration_path data/multi-zone-config.json \
        --source_video_path data/market-square.mp4 \
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
