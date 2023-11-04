<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/supervision/roboflow-supervision-banner.png?ik-sdk-version=javascript-1.4.3&updatedAt=1674062891088"
      >
    </a>
  </p>
</div>

## 👋 Hello

We write your reusable computer vision tools. Whether you need to load your dataset from your hard drive, draw detections on an image or video, or count how many detections are in a zone. You can count on us!

## 💻 Install

You can install `supervision` with pip in a
[**3.11>=Python>=3.8**](https://www.python.org/) environment.

!!! example "pip install (recommended)"

        ```bash
        pip install supervision
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

        # installation
        pip install -e "."
        ```

    === "poetry"

        ```bash
        # clone repository and navigate to root directory
        git clone https://github.com/roboflow/supervision.git
        cd supervision

        # setup python environment and activate it
        poetry env use python3.10
        poetry shell

        # dev installation
        poetry install

        ```
