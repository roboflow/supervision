# speed estimation

## 👋 hello

TODO

## 💻 install

- clone repository and navigate to example directory

  ```bash
  git clone https://github.com/roboflow/supervision.git
  cd supervision/examples/speed_estimation
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
  file, which is essential for the object detection process. This file contains the
  data that the model uses to identify objects in the video.
- `--source_video_path`: Required. The path to the source video file that will be
  analyzed. This is the input video on which traffic flow analysis will be performed.
- `--target_video_path` (optional): The path to save the output video with
  annotations. If not specified, the processed video will be displayed in real-time
  without being saved.
- `--confidence_threshold` (optional): Sets the confidence threshold for the YOLO
  model to filter detections. Default is `0.3`. This determines how confident the
  model should be to recognize an object in the video.
- `--iou_threshold` (optional): Specifies the IOU (Intersection Over Union) threshold
  for the model. Default is 0.7. This value is used to manage object detection
  accuracy, particularly in distinguishing between different objects.

## 📌 line configuration

TODO

## ⚙️ run

```bash
  python ultralytics_example.py \
  --source_weights_path yolov8x.pt \
  --lines_configuration_path data/vehicles-config.json \
  --source_video_path data/vehicles.mp4 \
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