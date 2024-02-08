# Manual Assembly Quality Assurance

[![Roboflow](https://raw.githubusercontent.com/roboflow-ai/notebooks/main/assets/badges/roboflow-blogpost.svg)](https://blog.roboflow.com/manual-assembly-qa/)

## 👋 hello

This example shows how to build a manual assembly quality assurance system with computer vision. The accompanying blog post walks through how to train a model to identify industrial parts. The code in this project lets you run the model with Inference and process predictions with supervision's tracking and smoothing. Predictions are visualized with supervision and OpenCV.

## 💻 install

- clone repository and navigate to example directory

  ```bash
  git clone https://github.com/roboflow/supervision.git
  cd supervision/examples/manual_assembly_quality_assurance
  ```

- setup python environment and activate it [optional]

  ```bash
  python3.10 -m venv venv
  source venv/bin/activate
  ```

- install required dependencies

  ```bash
  pip install -r requirements.txt
  ```

## 🛠️ script arguments

- `--roboflow_api_key` (optional): The API key for Roboflow services. If not provided
  directly, the script tries to fetch it from the `ROBOFLOW_API_KEY` environment
  variable. Follow [this guide](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key)
  to acquire your `API KEY`.
- `--model_id` (optional): Designates the Roboflow model ID to be used. The default
  value is `lego-assembly/2`. [Learn how to retrieve your model ID](https://docs.roboflow.com/api-reference/workspace-and-project-ids).

- `--order`: Required. The order in which the parts are assembled. This is used to
  ensure an assembler picks up pieces in the correct order. This should be a list of comma-separated strings. Each item in the list must be exactly equal to the class name associated with the object you want to identify. You can find your class names using the "Class Name" tab in the Roboflow model's "Model Info" section. For the Lego demo, the classes are `Orange 4x2 Brick`, `Green 4x2 Brick`, and `Blue 4x2 Brick`.

## ⚙️ run

  ```bash
    python app.py \
    --roboflow_api_key <ROBOFLOW API KEY> \
    --order "Orange 4x2 Brick,Green 4x2 Brick,Blue 4x2 Brick,Orange 4x2 Brick"
    --model_id "lego-assembly/2"
  ```

## © license

This demo integrates two main components, each with its own licensing:

- ultralytics: The object detection model used in this demo, YOLOv8, is distributed under the [AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

- supervision: The code that powers the manual assembly quality assurance logic in `app.py` is made with the Supervision library, which is licensed under the [MIT license](https://github.com/roboflow/supervision/blob/develop/LICENSE.md).
