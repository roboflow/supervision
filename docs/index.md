---
template: index.html
title: Supervision – Open Source Computer Vision Toolkit
description: Supervision is a lightweight open-source computer vision toolkit. Easily load datasets, draw detections, count objects, and analyze videos using Python.
comments: true
hide:
  - navigation
  - toc
---

<div class="md-typeset">
  <h1 align="center">Supervision – Open Source Computer Vision Toolkit</h1>
</div>

<div align="center" id="logo" style="padding-top: 1rem;">
  <a href="https://github.com/roboflow/supervision" target="_blank">
      <img
          width="850"
          src="https://media.roboflow.com/open-source/supervision/rf-supervision-banner.png?updatedAt=1678995927529"
          alt="Supervision: Open Source Computer Vision Toolkit Banner"
      >
  </a>
</div>

<style>
    #hello {
        margin: 0;
    }
</style>

## 👋 Hello

We build **reusable computer vision tools** to make your workflow faster and easier.  
Whether you need to **load datasets**, **visualize detections**, or **count objects in specific zones** — you can count on us!

<video controls>
    <source
        src="https://media.roboflow.com/traffic_analysis_result.mp4"
        type="video/mp4"
    >
</video>

---

## 💻 Install

You can install `supervision` in a
[**Python ≥ 3.9**](https://www.python.org/) environment.

!!! example "Installation"

    === "pip (recommended)"
        [![version](https://badge.fury.io/py/supervision.svg)](https://badge.fury.io/py/supervision)
        [![downloads](https://img.shields.io/pypi/dm/supervision)](https://pypistats.org/packages/supervision)
        [![license](https://img.shields.io/pypi/l/supervision)](https://github.com/roboflow/supervision/blob/main/LICENSE.md)
        [![python-version](https://img.shields.io/pypi/pyversions/supervision)](https://badge.fury.io/py/supervision)

        ```bash
        pip install supervision
        ```

    === "poetry"
        ```bash
        poetry add supervision
        ```

    === "uv"
        ```bash
        uv pip install supervision
        ```

        For uv projects:

        ```bash
        uv add supervision
        ```

    === "rye"
        ```bash
        rye add supervision
        ```

!!! example "conda/mamba install"
    === "conda"
        [![conda-recipe](https://img.shields.io/badge/recipe-supervision-green.svg)](https://anaconda.org/conda-forge/supervision)
        [![conda-downloads](https://img.shields.io/conda/dn/conda-forge/supervision.svg)](https://anaconda.org/conda-forge/supervision)
        [![conda-version](https://img.shields.io/conda/vn/conda-forge/supervision.svg)](https://anaconda.org/conda-forge/supervision)
        [![conda-platforms](https://img.shields.io/conda/pn/conda-forge/supervision.svg)](https://anaconda.org/conda-forge/supervision)

        ```bash
        conda install -c conda-forge supervision
        ```

    === "mamba"
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

---

## 📘 Documentation Overview

Explore essential topics to get started with Supervision:

- [Getting Started Guide](getting_started.md)
- [API Reference](reference/index.md)
- [Contributing Guidelines](https://github.com/roboflow/supervision/blob/main/CONTRIBUTING.md)
- [Examples Repository](https://github.com/roboflow/notebooks)
- [Changelog](https://github.com/roboflow/supervision/releases)

---

## 🚀 Quickstart

<div class="grid cards" markdown>

- **Detect and Annotate**

    ---

    Annotate predictions from various object detection and segmentation models.

    [:octicons-arrow-right-24: Tutorial](how_to/detect_and_annotate.md)

- **Track Objects**

    ---

    Enhance video analysis by implementing seamless object tracking.

    [:octicons-arrow-right-24: Tutorial](how_to/track_objects.md)

- **Detect Small Objects**

    ---

    Learn how to detect small objects in images effectively.

    [:octicons-arrow-right-24: Tutorial](how_to/detect_small_objects.md)

- **Count Objects Crossing Line**

    ---

    Accurately count and analyze objects crossing a predefined line.

    [:octicons-arrow-right-24: Notebook](https://supervision.roboflow.com/latest/notebooks/count-objects-crossing-the-line/)

- **Filter Objects in Zone**

    ---

    Master the techniques to selectively filter and focus on objects within a specific zone.

    [:octicons-arrow-right-24: Tutorial](how_to/filter_objects_in_zone.md)

- **Cheatsheet**

    ---

    Access a quick reference guide to the most common `supervision` functions.

    [:octicons-arrow-right-24: Cheatsheet](https://roboflow.github.io/cheatsheet-supervision/)
</div>

---

## 💬 Join the Community

Have questions or ideas? Join us:
- 💻 [GitHub Discussions](https://github.com/roboflow/supervision/discussions)
- 🐦 [Follow Roboflow on X (Twitter)](https://twitter.com/roboflow)
- 💬 [Discord Community](https://discord.gg/roboflow)
