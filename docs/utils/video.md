---
comments: true
---

# Video Utils

## New API: `Video`

The new `sv.Video` class provides a unified, extensible, and backend-agnostic interface for all video operations in Supervision. It supports static files, RTSP streams, and webcams, and can use either OpenCV or PyAV as a backend.

### Usage Examples

#### Get video info (file, RTSP, webcam)
```python
import supervision as sv

# static video
sv.Video("source.mp4").info

# video stream
sv.Video("rtsp://...").info

# webcam
sv.Video(0).info
```

#### Simple frame iteration (object is iterable)
```python
video = sv.Video("source.mp4")
for frame in video:
    ...
```

#### Advanced frame iteration (stride, sub-clip, on-the-fly resize)
```python
for frame in sv.Video("source.mp4").frames(stride=5, start=100, end=500, resolution_wh=(1280, 720)):
    ...
```

#### Process the video
```python
import cv2
import supervision as sv

def blur(frame, i):
    return cv2.GaussianBlur(frame, (11, 11), 0)

sv.Video("source.mp4").save(
    "blurred.mp4",
    callback=blur,
    show_progress=True
)
```

#### Overwrite target video parameters
```python
sv.Video("source.mp4").save(
    "timelapse.mp4",
    fps=60,
    callback=lambda f, i: f,
    show_progress=True
)
```

#### Complete manual control with explicit `VideoInfo`
```python
from supervision import Video, VideoInfo

src = Video("source.mp4")
target_info = VideoInfo(width=800, height=800, fps=24)

with src.sink("square.mp4", info=target_info) as sink:
    for f in src.frames():
        f = cv2.resize(f, target_info.resolution_wh)
        sink.write(f)
```

#### Multi-backend support (OpenCV, PyAV)
```python
video = sv.Video("source.mkv", backend="pyav")
video = sv.Video("source.mkv", backend="opencv")
```

---

## Deprecated API (to be removed in 5 releases)

The following classes and functions are deprecated. Please use the new `sv.Video` class for all new code. The old API is still available for backward compatibility, but will be removed in a future release.

- `VideoInfo` → use `sv.Video(...).info`
- `VideoSink` → use `sv.Video(...).sink()`
- `FPSMonitor` → no direct replacement (use time/perf_counter in your loop)
- `get_video_frames_generator` → use `sv.Video(...).frames()`
- `process_video` → use `sv.Video(...).save()`

:::supervision.utils.video.VideoInfo

:::supervision.utils.video.VideoSink

:::supervision.utils.video.FPSMonitor

:::supervision.utils.video.get_video_frames_generator

:::supervision.utils.video.process_video

---

## Backend Protocol

The new API is extensible via a backend protocol. You can implement your own backend by following the `Backend` protocol in `supervision/video.py`.
