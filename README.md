<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="100%"
        src="https://media.roboflow.com/open-source/supervision/rf-supervision-banner.png?updatedAt=1678995927529"
      >
    </a>
  </p>

  <br>

  <div align="center">
      <a href="https://youtube.com/roboflow">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/youtube.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949634652"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/roboflow-app.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949746649"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://www.linkedin.com/company/roboflow-ai/">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/linkedin.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633691"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://docs.roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/knowledge.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949634511"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://disuss.roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/forum.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633584"
            width="3%"
          />
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://blog.roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/blog.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633605"
            width="3%"
          />
      </a>
      </a>
  </div>

  <br>

[![version](https://badge.fury.io/py/supervision.svg)](https://badge.fury.io/py/supervision)
[![downloads](https://img.shields.io/pypi/dm/supervision)](https://pypistats.org/packages/supervision)
[![license](https://img.shields.io/pypi/l/supervision)](https://github.com/roboflow/supervision/blob/main/LICENSE.md)
[![python-version](https://img.shields.io/pypi/pyversions/supervision)](https://badge.fury.io/py/supervision)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-detect-and-count-objects-in-polygon-zone.ipynb)

</div>

## 👋 hello

We write your reusable computer vision tools. Whether you need to load your dataset from your hard drive, draw detections on an image or video, or count how many detections are in a zone. You can count on `supervision`! 🤝

## 💻 install

Pip install the supervision package in a
[**3.10>=Python>=3.7**](https://www.python.org/) environment.

```bash
pip install supervision
```

<details close>
<summary>install from source</summary>

```bash
# clone repository and navigate to root directory
git clone https://github.com/roboflow/supervision.git
cd supervision

# setup python environment and activate it
python3 -m venv venv
source venv/bin/activate

# install
pip install -e ".[dev]"
```

</details>

## 🔥 quickstart

### [detections processing](https://roboflow.github.io/supervision/detection/core/)

```python
>>> import supervision as sv
>>> from ultralytics import YOLO

>>> model = YOLO('yolov8s.pt')
>>> result = model(IMAGE)[0]
>>> detections = sv.Detections.from_yolov8(result)

>>> len(detections)
5
```

<details close>
<summary>👉 more detections utils</summary>
  
- Easily switch inference pipeline between supported object detection / instance segmentation models
  
    ```python
    >>> import supervision as sv
    >>> from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    >>> sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    >>> mask_generator = SamAutomaticMaskGenerator(sam)
    >>> sam_result = mask_generator.generate(IMAGE)
    >>> detections = sv.Detections.from_sam(sam_result=sam_result)
    ```
 
- [Advanced filtering](https://roboflow.github.io/supervision/quickstart/detections/)
  
    ```python
    >>> detections = detections[detections.class_id == 0]
    >>> detections = detections[detections.confidence > 0.5]
    >>> detections = detections[detections.area > 1000]
    ```
  
- Image annotation
  
    ```python
    >>> import supervision as sv

    >>> box_annotator = sv.BoxAnnotator()
    >>> annotated_frame = box_annotator.annotate(
    ...     scene=IMAGE,
    ...     detections=detections
    ... )
    ```
  
</details>

### [datasets processing](https://roboflow.github.io/supervision/dataset/core/)

```python
>>> import supervision as sv

>>> dataset = sv.DetectionDataset.from_yolo(
...     images_directory_path='...',
...     annotations_directory_path='...',
...     data_yaml_path='...'
... )

>>> dataset.classes
['dog', 'person']

>>> len(dataset)
1000
```

<details close>
<summary>👉 more dataset utils</summary>

- Load object detection / instance segmentation datasets in one of supported formats

    ```python
    >>> dataset = sv.DetectionDataset.from_yolo(
    ...     images_directory_path='...',
    ...     annotations_directory_path='...',
    ...     data_yaml_path='...'
    ... )

    >>> dataset = sv.DetectionDataset.from_pascal_voc(
    ...     images_directory_path='...',
    ...     annotations_directory_path='...'
    ... )
    ```
  
- Loop over dataset entries

    ```python
    >>> for image, labels in dataset:
    ...     print(labels.xyxy)

    array([[404.      , 719.      , 538.      , 884.5     ],
           [155.      , 497.      , 404.      , 833.5     ],
           [ 20.154999, 347.825   , 416.125   , 915.895   ]], dtype=float32)
    ```
  
- Split dataset for training, testing and validation
  
    ```python
    >>> train_dataset, test_dataset = dataset.split(split_ratio=0.7)
    >>> test_dataset, valid_dataset = test_dataset.split(split_ratio=0.5)
  
    >>> len(train_dataset), len(test_dataset), len(valid_dataset)
    (700, 150, 150)
    ```
  
- Save object detection / instance segmentation datasets in one of supported formats
  
    ```python
    >>> dataset.as_yolo(
    ...     images_directory_path='...',
    ...     annotations_directory_path='...',
    ...     data_yaml_path='...'
    ... )

    >>> dataset.as_pascal_voc(
    ...     images_directory_path='...',
    ...     annotations_directory_path='...'
    ... )
    ```
  
- Convert labels between suppoted formats
  
    ```python
    >>> sv.DetectionDataset.from_yolo(
    ...     images_directory_path='...',
    ...     annotations_directory_path='...',
    ...     data_yaml_path='...'
    ... ).as_pascal_voc(
    ...     images_directory_path='...',
    ...     annotations_directory_path='...'
    ... )
    ```

</details>

## 🎬 tutorials

🔥 [Subscribe](https://www.youtube.com/@Roboflow) and stay up to date with the latest changes to the Supervision library. 

<p align="center">
    <a href="https://youtu.be/l_kf9CfZ_8M">
        <img 
            width="90%"
            src="https://user-images.githubusercontent.com/26109316/217950212-311de186-1862-4b4c-a86e-89cafd68b233.jpg" 
            alt="count-objects-in-zone"
        >
    </a>
</p>

<p align="center">
    <a href="https://youtu.be/OS5qI9YBkfk">
        <img
            width="90%"
            src="https://user-images.githubusercontent.com/26109316/213702005-ddd568f0-b902-46c2-9af9-e6ff33db23bf.jpg" 
            alt="count-objects-crossing-line"
        >
    </a>
</p>

## 📚 documentation

Curious how Supervision can help you solve problems on your project? Visit our [documentation](https://roboflow.github.io/supervision) page!

## 🏆 contribution

We love your input! Please see our [contributing guide](https://github.com/roboflow/supervision/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!
