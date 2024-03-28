---
template: index.html
comments: true
hide:
  - navigation
  - toc
---

<div class="md-typeset">
  <h1></h1>
</div>

<div align="center" id="logo">
  <a align="center" href="" target="_blank">
      <img width="850"
          src="https://media.roboflow.com/open-source/supervision/rf-supervision-banner.png?updatedAt=1678995927529">
  </a>
</div>

## 👋 Hello

We write your reusable computer vision tools. Whether you need to load your dataset from your hard drive, draw detections on an image or video, or count how many detections are in a zone. You can count on us!

<video controls>
    <source
        src="https://media.roboflow.com/traffic_analysis_result.mp4"
        type="video/mp4"
    >
</video>

## 💻 Install

You can install `supervision` with pip in a
[**Python>=3.8**](https://www.python.org/) environment.

!!! success "Installation"

    !!! example "Pip installation (recommended)"

        === "Headless"
            The headless installation of `supervision` is designed for environments where graphical user interfaces (GUI) are not needed, making it more lightweight and suitable for server-side applications.

            ```bash
            pip install supervision
            ```

        === "Desktop"
            The desktop installation of `supervision` is designed with GUI support. This version includes the GUI components of OpenCV, allowing you to display images and videos on the screen.

            ```bash
            pip install "supervision[desktop]"
            ```

    !!! example "Conda/Mamba installation"

        === "conda"
            The Conda installation of `supervision` is designed for those who prefer using Conda as their package manager. It's especially useful for managing complex dependencies and environments.

            [![Conda Recipe](https://img.shields.io/badge/recipe-supervision-green.svg)](https://anaconda.org/conda-forge/supervision) [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/supervision.svg)](https://anaconda.org/conda-forge/supervision) [![Conda Version](https://img.shields.io/conda/vn/conda-forge/supervision.svg)](https://anaconda.org/conda-forge/supervision) [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/supervision.svg)](https://anaconda.org/conda-forge/supervision)


            ```bash
            conda install -c conda-forge supervision
            ```

        === "mamba"
            The Mamba installation of `supervision` is ideal for those who prefer using Mamba as their package manager.

            [![Mamba Recipe](https://img.shields.io/badge/recipe-supervision-green.svg)](https://anaconda.org/conda-forge/supervision) [![Mamba Downloads](https://img.shields.io/conda/dn/conda-forge/supervision.svg)](https://anaconda.org/conda-forge/supervision) [![Mamba Version](https://img.shields.io/conda/vn/conda-forge/supervision.svg)](https://anaconda.org/conda-forge/supervision) [![Mamba Platforms](https://img.shields.io/conda/pn/conda-forge/supervision.svg)](https://anaconda.org/conda-forge/supervision)

            ```bash
            mamba install -c conda-forge supervision
            ```

    !!! example "git clone (for development)"

        === "virtualenv"

            ```bash
            # clone repository and navigate to root directory
            git clone https://github.com/roboflow/supervision.git
            cd supervision

            # setup python environment and activate it
            python3 -m venv venv
            source venv/bin/activate
            pip install --upgrade pip

            # headless install
            pip install -e "."

            # desktop install
            pip install -e ".[desktop]"
            ```

        === "poetry"

            ```bash
            # clone repository and navigate to root directory
            git clone https://github.com/roboflow/supervision.git
            cd supervision

            # setup python environment and activate it
            poetry env use python3.10
            poetry shell

            # headless install
            poetry install

            # desktop install
            poetry install --extras "desktop"
            ```


## 🚀 Quickstart

<div class="grid cards" markdown>

-   __Detect and Annotate__

    ---

    Annotate predictions from a range of object detection and segmentation models

    [:octicons-arrow-right-24: Tutorial](how_to/detect_and_annotate.md)

-   __Track Objects__

    ---

    Discover how to enhance video analysis by implementing seamless object tracking

    [:octicons-arrow-right-24: Tutorial](how_to/track_objects.md)

-   __Detect Small Objects__

    ---

    Learn how to detect small objects in images

    [:octicons-arrow-right-24: Tutorial](how_to/detect_small_objects.md)

-   > __Count Objects Crossing Line__

    ---

    Explore methods to accurately count and analyze objects crossing a predefined line

-   > __Filter Objects in Zone__

    ---

    Master the techniques to selectively filter and focus on objects within a specific zone

</div>
