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
  
## 🛠️ script arguments

  - `--source_weights_path` (optional): The path to the YOLO model's weights file.
    Defaults to `"yolov8x.pt"` if not specified.

  - `--zone_configuration_path`: Specifies the path to the JSON file containing zone
    configurations. This file defines the polygonal areas in the video where objects will
    be counted.
  - `--source_video_path`: The path to the source video file that will be analyzed.

## ⚙️ run

```bash
python ultralytics_example.py \
  --source_weights_path yolov8m.pt \
  --zone_configuration_path data/time-in-zone-video.json \
  --source_video_path data/time-in-zone-video.mp4 \
  --confidence_threshold 0.5
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
