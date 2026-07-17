---
comments: true
description: Migrate Supervision installations after OpenCV becomes an ambient optional backend: use the included fallback by default or select one compatible OpenCV wheel for your application.
date_modified: 2026-07-17
---

# Migrate to Supervision Without an OpenCV Dependency

Supervision no longer installs OpenCV or offers an OpenCV extra. A standard installation includes the NumPy, Pillow, SciPy, and PyAV fallback needed by Supervision's image, drawing, and file-video APIs. When a compatible `cv2` is already installed, Supervision selects it once when the process imports the package.

## Keep the default fallback

Install Supervision normally when your application does not otherwise require OpenCV:

```bash
pip install supervision
```

The fallback keeps Supervision's documented APIs operational. Some text and anti-aliased drawing pixels can differ from OpenCV, so use the same backend while validating image-level baselines.

## Prefer OpenCV behavior

Install exactly one OpenCV wheel family when your application relies on OpenCV outside Supervision or needs its native behavior:

```bash
# Servers and containers without OpenCV GUI modules
pip install opencv-python-headless supervision

# Desktop applications that need OpenCV GUI modules
pip install opencv-python supervision
```

Do not install both `opencv-python` and `opencv-python-headless`. If another dependency, such as a model runtime, already provides a compatible `cv2`, keep that installation instead of adding a second wheel family.

## Verify the selected backend

Backend selection happens at import time and lasts for the process lifetime. Run this command in a fresh Python process after changing dependencies:

```bash
python -c "from supervision import _cv2; print(_cv2.BACKEND_NAME)"
```

It prints `fallback` without OpenCV and the OpenCV backend name when `cv2` is available. `_cv2` is private; use this command only as an installation diagnostic, not as application API.

## Capture webcams yourself

Supervision's video helpers support file paths through either backend. Live camera capture remains application-owned, so install your chosen OpenCV wheel if you use `cv2.VideoCapture(0)`:

```python
import cv2
import supervision as sv

capture = cv2.VideoCapture(0)
annotator = sv.BoxAnnotator()
```

## Roll back an upgrade

If a downstream image baseline requires the pre-migration package behavior, pin Supervision below the first release that removes the OpenCV dependency, then plan a backend-specific migration separately:

```bash
pip install "supervision<0.30.0"
```
