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

[notebooks](https://github.com/roboflow/notebooks) | [inference](https://github.com/roboflow/inference) | [autodistill](https://github.com/autodistill/autodistill) | [collect](https://github.com/roboflow/roboflow-collect)

  <br>

[![version](https://badge.fury.io/py/supervision.svg)](https://badge.fury.io/py/supervision)
[![downloads](https://img.shields.io/pypi/dm/supervision)](https://pypistats.org/packages/supervision)
[![license](https://img.shields.io/pypi/l/supervision)](https://github.com/roboflow/supervision/blob/main/LICENSE.md)
[![python-version](https://img.shields.io/pypi/pyversions/supervision)](https://badge.fury.io/py/supervision)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/supervision/blob/main/demo.ipynb)
[![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Roboflow/Annotators)
[![Discord](https://img.shields.io/discord/1159501506232451173)](https://discord.gg/GbfgXGJ8Bk)

</div>

## 👋 hello

**We write your reusable computer vision tools.** Whether you need to load your dataset from your hard drive, draw detections on an image or video, or count how many detections are in a zone. You can count on us! 🤝

## 💻 install

Pip install the supervision package in a
[**Python>=3.8**](https://www.python.org/) environment.

```bash
pip install supervision
```

Read more about desktop, headless, and local installation in our [guide](https://roboflow.github.io/supervision/).

## 🔥 quickstart

### models

Supervision was designed to be model agnostic. Just plug in any classification, detection, or segmentation model. For your convenience, we have created [connectors](https://supervision.roboflow.com/detection/core/#detections) for the most popular libraries like Ultralytics, Transformers, or MMDetection.

```python
>>> import cv2
>>> import supervision as sv
>>> from ultralytics import YOLO

>>> image = cv2.imread(...)
>>> model = YOLO('yolov8s.pt')
>>> result = model(image)[0]
>>> detections = sv.Detections.from_ultralytics(result)

>>> len(detections)
5
```

<details>
<summary>👉 more model connectors</summary>

- inference

    Running with [Inference](https://github.com/roboflow/inference) requires a [Roboflow API KEY](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key).

    ```python
    >>> import cv2
    >>> import supervision as sv
    >>> from inference.models.utils import get_roboflow_model

    >>> image = cv2.imread(...)
    >>> model = get_roboflow_model(model_id="yolov8s-640", api_key=<ROBOFLOW API KEY>)
    >>> result = model.infer(image)[0]
    >>> detections = sv.Detections.from_inference(result)

    >>> len(detections)
    >>> 5

    ```

</details>

### annotators

Supervision offers a wide range of highly customizable [annotators](https://supervision.roboflow.com/annotators/), allowing you to compose the perfect visualization for your use case.

```python
>>> import cv2
>>> import supervision as sv

>>> image = cv2.imread(...)
>>> detections = sv.Detections(...)

>>> bounding_box_annotator = sv.BoundingBoxAnnotator()
>>> annotated_frame = bounding_box_annotator.annotate(
...     scene=image.copy(),
...     detections=detections
... )
```

https://github.com/roboflow/supervision/assets/26109316/691e219c-0565-4403-9218-ab5644f39bce

### datasets

Supervision provides a set of [utils](https://supervision.roboflow.com/datasets/) that allow you to load, split, merge, and save datasets in one of the supported formats.

```python
>>> import supervision as sv

>>> dataset = sv.DetectionDataset.from_yolo(
...     images_directory_path=...,
...     annotations_directory_path=...,
...     data_yaml_path=...
... )

>>> dataset.classes
['dog', 'person']

>>> len(dataset)
1000
```

<details close>
<summary>👉 more dataset utils</summary>

- load

  ```python
  >>> dataset = sv.DetectionDataset.from_yolo(
  ...     images_directory_path=...,
  ...     annotations_directory_path=...,
  ...     data_yaml_path=...
  ... )

  >>> dataset = sv.DetectionDataset.from_pascal_voc(
  ...     images_directory_path=...,
  ...     annotations_directory_path=...
  ... )

  >>> dataset = sv.DetectionDataset.from_coco(
  ...     images_directory_path=...,
  ...     annotations_path=...
  ... )
  ```

- split

  ```python
  >>> train_dataset, test_dataset = dataset.split(split_ratio=0.7)
  >>> test_dataset, valid_dataset = test_dataset.split(split_ratio=0.5)

  >>> len(train_dataset), len(test_dataset), len(valid_dataset)
  (700, 150, 150)
  ```

- merge

  ```python
  >>> ds_1 = sv.DetectionDataset(...)
  >>> len(ds_1)
  100
  >>> ds_1.classes
  ['dog', 'person']

  >>> ds_2 = sv.DetectionDataset(...)
  >>> len(ds_2)
  200
  >>> ds_2.classes
  ['cat']

  >>> ds_merged = sv.DetectionDataset.merge([ds_1, ds_2])
  >>> len(ds_merged)
  300
  >>> ds_merged.classes
  ['cat', 'dog', 'person']
  ```

- save

  ```python
  >>> dataset.as_yolo(
  ...     images_directory_path=...,
  ...     annotations_directory_path=...,
  ...     data_yaml_path=...
  ... )

  >>> dataset.as_pascal_voc(
  ...     images_directory_path=...,
  ...     annotations_directory_path=...
  ... )

  >>> dataset.as_coco(
  ...     images_directory_path=...,
  ...     annotations_path=...
  ... )
  ```

- convert

  ```python
  >>> sv.DetectionDataset.from_yolo(
  ...     images_directory_path=...,
  ...     annotations_directory_path=...,
  ...     data_yaml_path=...
  ... ).as_pascal_voc(
  ...     images_directory_path=...,
  ...     annotations_directory_path=...
  ... )
  ```

</details>

## 🎬 tutorials

<p align="left">
<a href="https://youtu.be/4Q3ut7vqD5o" title="Traffic Analysis with YOLOv8 and ByteTrack - Vehicle Detection and Tracking"><img src="https://github.com/roboflow/supervision/assets/26109316/54afdf1c-218c-4451-8f12-627fb85f1682" alt="Traffic Analysis with YOLOv8 and ByteTrack - Vehicle Detection and Tracking" width="300px" align="left" /></a>
<a href="https://youtu.be/4Q3ut7vqD5o" title="Traffic Analysis with YOLOv8 and ByteTrack - Vehicle Detection and Tracking"><strong>Traffic Analysis with YOLOv8 and ByteTrack - Vehicle Detection and Tracking</strong></a>
<div><strong>Created: 6 Sep 2023</strong> | <strong>Updated: 6 Sep 2023</strong></div>
<br/> In this video, we explore real-time traffic analysis using YOLOv8 and ByteTrack to detect and track vehicles on aerial images. Harnessing the power of Python and Supervision, we delve deep into assigning cars to specific entry zones and understanding their direction of movement. By visualizing their paths, we gain insights into traffic flow across bustling roundabouts... </p>

<br/>

<p align="left">
<a href="https://youtu.be/D-D6ZmadzPE" title="SAM - Segment Anything Model by Meta AI: Complete Guide"><img src="https://github.com/SkalskiP/SkalskiP/assets/26109316/6913ff11-53c6-4341-8d90-eaff3023c3fd" alt="SAM - Segment Anything Model by Meta AI: Complete Guide" width="300px" align="left" /></a>
<a href="https://youtu.be/D-D6ZmadzPE" title="SAM - Segment Anything Model by Meta AI: Complete Guide"><strong>SAM - Segment Anything Model by Meta AI: Complete Guide</strong></a>
<div><strong>Created: 11 Apr 2023</strong> | <strong>Updated: 11 Apr 2023</strong></div>
<br/> Discover the incredible potential of Meta AI's Segment Anything Model (SAM)! We dive into SAM, an efficient and promptable model for image segmentation, which has revolutionized computer vision tasks. With over 1 billion masks on 11M licensed and privacy-respecting images, SAM's zero-shot performance is often competitive with or even superior to prior fully supervised results... </p>

## 💜 built with supervision

Did you build something cool using supervision? [Let us know!](https://github.com/roboflow/supervision/discussions/categories/built-with-supervision)

https://user-images.githubusercontent.com/26109316/207858600-ee862b22-0353-440b-ad85-caa0c4777904.mp4

https://github.com/roboflow/supervision/assets/26109316/c9436828-9fbf-4c25-ae8c-60e9c81b3900

https://github.com/roboflow/supervision/assets/26109316/f84db7b5-79e2-4142-a1da-64daa43ce667

## 📚 documentation

Visit our [documentation](https://roboflow.github.io/supervision) page to learn how supervision can help you build computer vision applications faster and more reliably.

## 🏆 contribution

We love your input! Please see our [contributing guide](https://github.com/roboflow/supervision/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!

<p align="center">
    <a href="https://github.com/roboflow/supervision/graphs/contributors">
      <img src="https://contrib.rocks/image?repo=roboflow/supervision" />
    </a>
</p>

<br>

<div align="center">

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
</div>
