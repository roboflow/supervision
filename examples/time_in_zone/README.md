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

- download video from youtube

  - `--url`: The full URL of the YouTube video you wish to download.
  - `--output_path` (optional): Specifies the directory where the video will be saved.
  - `--file_name` (optional): Sets the name of the saved video file.


- stream video locally

  - `--video_directory`: Directory containing video files to stream.
  - `--number_of_streams`: Number of video files to stream.

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