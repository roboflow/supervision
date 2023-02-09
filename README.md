<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/supervision/roboflow-supervision-banner.png?ik-sdk-version=javascript-1.4.3&updatedAt=1674062891088"
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

</div>

## 👋 hello

A set of easy-to-use utils that will come in handy in any Computer Vision project. **Supervision** is still in 
pre-release stage. 🚧 Keep your eyes open for potential bugs and be aware that at this stage our API is still fluid 
and may change.

## 💻 install

Pip install the supervision package in a
[**3.10>=Python>=3.7**](https://www.python.org/) environment.

```bash
pip install supervision
```

<details close>
<summary>Install from source</summary>

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

## 📖 documentation

Curious how Supervision can help you solve problems on your project? Visit our [documentation](https://roboflow.github.io/supervision) page!

## 🎬 videos

Learn how to use YOLOv8, ByteTrack and **Supervision** to detect, track and count objects. 🔥
[Subscribe](https://www.youtube.com/@Roboflow), and stay up to date with our latest YouTube videos!

<p align="center">
    <a href="https://youtu.be/l_kf9CfZ_8M">
        <img src="https://user-images.githubusercontent.com/26109316/217950212-311de186-1862-4b4c-a86e-89cafd68b233.jpg" alt="count-objects-in-zone">
    </a>
</p>

<p align="center">
    <a href="https://youtu.be/OS5qI9YBkfk">
        <img src="https://user-images.githubusercontent.com/26109316/213702005-ddd568f0-b902-46c2-9af9-e6ff33db23bf.jpg" alt="count-objects-crossing-line">
    </a>
</p>

## 🧹 code quality 

We provide two handy commands inside the `Makefile`, namely:

- `make style` to format the code
- `make check_code_quality` to check code quality (PEP8 basically)

So far, **there is no types checking with mypy**. See [issue](https://github.com/roboflow-ai/template-python/issues/4). 

## 🧪 tests 

[`pytests`](https://docs.pytest.org/en/7.1.x/) is used to run our tests.
