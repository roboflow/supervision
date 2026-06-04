---
template: index.html
comments: true
hide:
  - navigation
  - toc
description: Open-source Python library providing computer vision tools for annotating detections, tracking objects, counting in zones, and processing datasets.
---

<div class="md-typeset">
  <h1></h1>
</div>

<div align="center" id="logo" style="padding-top: 1rem;">
  <a align="center" href="" target="_blank">
      <img width="850"
          src="https://media.roboflow.com/open-source/supervision/rf-supervision-banner.png?updatedAt=1678995927529">
  </a>
</div>

<style>
    #hello {
        margin: 0;
    }
</style>

## What is Supervision?

Supervision is an open-source Python library by Roboflow for building computer vision applications. It provides a unified `Detections` object with converters for supported outputs from Ultralytics, Roboflow Inference, Transformers, SAM, Detectron2, MMDetection, YOLO-NAS, PaddleDet, NCNN, Azure AI Vision, and VLM parsers.

With Supervision you can annotate images and video with bounding boxes, masks, and labels; track objects across frames with persistent IDs; count and filter detections inside polygon zones; load and convert datasets between YOLO, COCO, and Pascal VOC formats; and benchmark model performance with mAP and confusion matrices.

Supervision is MIT licensed, has 38,000+ GitHub stars, and over 1 million monthly PyPI downloads. It is developed in public on GitHub for production computer vision workflows.

## 👋 Hello

We write your reusable computer vision tools. Whether you need to load your dataset from your hard drive, draw detections on an image or video, or count how many detections are in a zone. You can count on us!

<video controls>
    <source
        src="https://media.roboflow.com/traffic_analysis_result.mp4"
        type="video/mp4"
    >
</video>

## 💻 Install

You can install `supervision` in a
[**Python>=3.9**](https://www.python.org/) environment.

!!! example "Installation"

    === "pip (recommended)"

        [![version](https://badge.fury.io/py/supervision.svg)](https://badge.fury.io/py/supervision)
        [![downloads](https://img.shields.io/pypi/dm/supervision)](https://pypistats.org/packages/supervision)
        [![license](https://img.shields.io/pypi/l/supervision)](../LICENSE.md)
        [![python-version](https://img.shields.io/pypi/pyversions/supervision)](https://badge.fury.io/py/supervision)

        Supervision requires OpenCV. Choose the variant that best fits your needs:

        ⚠️ **Important**: Only install one OpenCV variant at a time. The different `opencv-python` packages (standard, contrib, headless, etc.) are mutually exclusive and cannot coexist in the same environment.

        ```bash
        # Recommended for servers (no GUI)
        pip install supervision[headless]
        ```

        ```bash
        # For desktop applications (includes GUI support)
        pip install supervision[desktop]
        ```

        ```bash
        # For desktop with extra modules (e.g., CSRT tracker)
        pip install supervision[desktop-contrib]
        ```

        ```bash
        # For servers with extra modules (no GUI)
        pip install supervision[headless-contrib]
        ```

        If you already have OpenCV installed:

        ```bash
        pip install supervision
        ```

    === "poetry"

        [![version](https://badge.fury.io/py/supervision.svg)](https://badge.fury.io/py/supervision)
        [![downloads](https://img.shields.io/pypi/dm/supervision)](https://pypistats.org/packages/supervision)
        [![license](https://img.shields.io/pypi/l/supervision)](../LICENSE.md)
        [![python-version](https://img.shields.io/pypi/pyversions/supervision)](https://badge.fury.io/py/supervision)

        ```bash
        # With headless OpenCV (recommended for servers)
        poetry add supervision[headless]
        ```

        ```bash
        # With desktop OpenCV (includes GUI)
        poetry add supervision[desktop]
        ```

        ```bash
        # For desktop with extra modules (e.g., CSRT tracker)
        poetry add supervision[desktop-contrib]
        ```

        ```bash
        # For servers with extra modules (no GUI)
        poetry add supervision[headless-contrib]
        ```

    === "uv"

        [![version](https://badge.fury.io/py/supervision.svg)](https://badge.fury.io/py/supervision)
        [![downloads](https://img.shields.io/pypi/dm/supervision)](https://pypistats.org/packages/supervision)
        [![license](https://img.shields.io/pypi/l/supervision)](../LICENSE.md)
        [![python-version](https://img.shields.io/pypi/pyversions/supervision)](https://badge.fury.io/py/supervision)

        ```bash
        # With headless OpenCV (recommended for servers)
        uv pip install supervision[headless]
        ```

        ```bash
        # For desktop applications (includes GUI support)
        uv pip install supervision[desktop]
        ```

        ```bash
        # For desktop with extra modules (e.g., CSRT tracker)
        uv pip install supervision[desktop-contrib]
        ```

        ```bash
        # For servers with extra modules (no GUI)
        uv pip install supervision[headless-contrib]
        ```

        If you already have OpenCV installed:

        ```bash
        uv pip install supervision
        ```

        For uv projects:

        ```bash
        uv add supervision --extra headless
        ```

    === "rye"

        [![version](https://badge.fury.io/py/supervision.svg)](https://badge.fury.io/py/supervision)
        [![downloads](https://img.shields.io/pypi/dm/supervision)](https://pypistats.org/packages/supervision)
        [![license](https://img.shields.io/pypi/l/supervision)](../LICENSE.md)
        [![python-version](https://img.shields.io/pypi/pyversions/supervision)](https://badge.fury.io/py/supervision)

        ```bash
        # With headless OpenCV (recommended for servers)
        rye add supervision --features headless
        ```

        ```bash
        # For desktop applications (includes GUI support)
        rye add supervision --features desktop
        ```

        ```bash
        # For desktop with extra modules (e.g., CSRT tracker)
        rye add supervision --features desktop-contrib
        ```

        ```bash
        # For servers with extra modules (no GUI)
        rye add supervision --features headless-contrib
        ```

!!! example "conda/mamba install"

    === "conda"

        [![conda-recipe](https://img.shields.io/badge/recipe-supervision-green.svg)](https://anaconda.org/conda-forge/supervision) [![conda-downloads](https://img.shields.io/conda/dn/conda-forge/supervision.svg)](https://anaconda.org/conda-forge/supervision) [![conda-version](https://img.shields.io/conda/vn/conda-forge/supervision.svg)](https://anaconda.org/conda-forge/supervision) [![conda-platforms](https://img.shields.io/conda/pn/conda-forge/supervision.svg)](https://anaconda.org/conda-forge/supervision)

        ```bash
        conda install -c conda-forge supervision
        ```

    === "mamba"

        [![mamba-recipe](https://img.shields.io/badge/recipe-supervision-green.svg)](https://anaconda.org/conda-forge/supervision) [![mamba-downloads](https://img.shields.io/conda/dn/conda-forge/supervision.svg)](https://anaconda.org/conda-forge/supervision) [![mamba-version](https://img.shields.io/conda/vn/conda-forge/supervision.svg)](https://anaconda.org/conda-forge/supervision) [![mamba-platforms](https://img.shields.io/conda/pn/conda-forge/supervision.svg)](https://anaconda.org/conda-forge/supervision)

        ```bash
        mamba install -c conda-forge supervision
        ```

!!! example "git clone (for development)"

    === "virtualenv"

        ```bash
        # clone repository and navigate to root directory
        git clone --depth 1 -b develop https://github.com/roboflow/supervision.git
        cd supervision

        # setup python environment and activate it
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip

        # installation
        pip install -e "."
        ```

    === "uv"

        ```bash
        # clone repository and navigate to root directory
        git clone --depth 1 -b develop https://github.com/roboflow/supervision.git
        cd supervision

        # setup python environment and activate it
        uv venv
        source .venv/bin/activate

        # installation
        uv pip install -r pyproject.toml -e . --all-extras

        ```

## 🚀 Quickstart

<div class="grid cards" markdown>

- **Detect and Annotate**

    ---

    Annotate predictions from a range of object detection and segmentation models

    [:octicons-arrow-right-24: Tutorial](how_to/detect_and_annotate.md)

- **Track Objects**

    ---

    Discover how to enhance video analysis by implementing seamless object tracking

    [:octicons-arrow-right-24: Tutorial](how_to/track_objects.md)

- **Detect Small Objects**

    ---

    Learn how to detect small objects in images

    [:octicons-arrow-right-24: Tutorial](how_to/detect_small_objects.md)

- **Count Objects Crossing Line**

    ---

    Explore methods to accurately count and analyze objects crossing a predefined line

    [:octicons-arrow-right-24: Notebook](https://supervision.roboflow.com/latest/notebooks/count-objects-crossing-the-line/)

- > **Filter Objects in Zone**

    ---

    Master the techniques to selectively filter and focus on objects within a specific zone

- **Cheatsheet**

    ---

    Access a quick reference guide to the most common `supervision` functions

    [:octicons-arrow-right-24: Cheatsheet](https://roboflow.github.io/cheatsheet-supervision/)

</div>
