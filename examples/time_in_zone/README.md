# time in zone

## 👋 hello

TODO

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
  
## 🛠 scripts
  
### `download_from_youtube`

- `--url`: The full URL of the YouTube video you wish to download.
- `--output_path` (optional): Specifies the directory where the video will be saved.
- `--file_name` (optional): Sets the name of the saved video file.

```bash
python scripts/download_from_youtube.py \
--url "https://youtu.be/8zyEwAa50Q" \
--output_path "data/checkout" \
--file_name "video.mp4"
```

```bash
python scripts/download_from_youtube.py \
--url "https://youtu.be/MNn9qKG2UFI" \
--output_path "data/traffic" \
--file_name "video.mp4"
```

### `stream_from_file`

- `--video_directory`: Directory containing video files to stream.
- `--number_of_streams`: Number of video files to stream.

```bash
python scripts/stream_from_file.py \
--video_directory "data/checkout" \
--number_of_streams 1
```

```bash
python scripts/stream_from_file.py \
--video_directory "data/traffic" \
--number_of_streams 1
```

## 🎬 video & stream processing

### `inference_file_example`

  - `--zone_configuration_path`: Path to the zone configuration JSON file.
  - `--source_video_path`: Path to the source video file.
  - `--model_id`: Roboflow model ID.
  - `--classes`: List of class IDs to track. If empty, all classes are tracked.
  - `--confidence_threshold`: Confidence level for detections (`0` to `1`). Default is `0.3`.
  - `--iou_threshold`: IOU threshold for non-max suppression. Default is `0.7`.

### `inference_stream_example`

  - `--zone_configuration_path`: Path to the zone configuration JSON file.
  - `--rtsp_url`: Complete RTSP URL for the video stream.
  - `--model_id`: Roboflow model ID.
  - `--classes`: List of class IDs to track. If empty, all classes are tracked.
  - `--confidence_threshold`: Confidence level for detections (`0` to `1`). Default is `0.3`.
  - `--iou_threshold`: IOU threshold for non-max suppression. Default is `0.7`.

### `inference_pipeline_stream_example`

  - `--zone_configuration_path`: Path to the zone configuration JSON file.
  - `--rtsp_url`: Complete RTSP URL for the video stream.
  - `--model_id`: Roboflow model ID.
  - `--classes`: List of class IDs to track. If empty, all classes are tracked.
  - `--confidence_threshold`: Confidence level for detections (`0` to `1`). Default is `0.3`.
  - `--iou_threshold`: IOU threshold for non-max suppression. Default is `0.7`.

### `ultralytics_file_example`

  - `--zone_configuration_path`: Path to the zone configuration JSON file.
  - `--source_video_path`: Path to the source video file.
  - `--weights`: Path to the model weights file. Default is `'yolov8s.pt'`.
  - `--device`: Computation device (`'cpu'`, `'mps'` or `'cuda'`). Default is `'cpu'`.
  - `--classes`: List of class IDs to track. If empty, all classes are tracked.
  - `--confidence_threshold`: Confidence level for detections (`0` to `1`). Default is `0.3`.
  - `--iou_threshold`: IOU threshold for non-max suppression. Default is `0.7`.

### `ultralytics_stream_example`

  - `--zone_configuration_path`: Path to the zone configuration JSON file.
  - `--rtsp_url`: Complete RTSP URL for the video stream.
  - `--weights`: Path to the model weights file. Default is `'yolov8s.pt'`.
  - `--device`: Computation device (`'cpu'`, `'mps'` or `'cuda'`). Default is `'cpu'`.
  - `--classes`: List of class IDs to track. If empty, all classes are tracked.
  - `--confidence_threshold`: Confidence level for detections (`0` to `1`). Default is `0.3`.
  - `--iou_threshold`: IOU threshold for non-max suppression. Default is `0.7`.

### `ultralytics_pipeline_stream_example`

  - `--zone_configuration_path`: Path to the zone configuration JSON file.
  - `--rtsp_url`: Complete RTSP URL for the video stream.
  - `--weights`: Path to the model weights file. Default is `'yolov8s.pt'`.
  - `--device`: Computation device (`'cpu'`, `'mps'` or `'cuda'`). Default is `'cpu'`.
  - `--classes`: List of class IDs to track. If empty, all classes are tracked.
  - `--confidence_threshold`: Confidence level for detections (`0` to `1`). Default is `0.3`.
  - `--iou_threshold`: IOU threshold for non-max suppression. Default is `0.7`.

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
