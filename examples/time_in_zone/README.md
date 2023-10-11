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
  
- download `people_walking.mp4` file

    ```bash
    ./setup.sh
    ```

## ⚙️ run

```bash
python script.py \
--source_weights_path yolov8x.pt \
--source_video_path data/people_walking.mp4 \
--confidence_threshold 0.3 \
--iou_threshold 0.5 \
--target_video_path data/people_walking_result.mp4
```