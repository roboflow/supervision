---
comments: true
---

# Video API (v2)

The new unified video interface lives in `supervision.Video` and is **backend-agnostic**. It supersedes the helpers previously found in `supervision.utils.video`.

## Quick-start

```python
import supervision as sv

# Static video
video = sv.Video("example.mp4")
print(video.info)

# Iterate with stride & resize
for frame in video.frames(stride=5, resolution_wh=(1280, 720)):
    ...

# Process and save
video.save(
    "blurred.mp4",
    callback=lambda f, i: cv2.GaussianBlur(f, (11, 11), 0),
    show_progress=True,
)
```

## `sv.Video` reference

::: supervision.video.Video

## `sv.VideoInfo` reference

::: supervision.video.common.VideoInfo

## Backends

`Video` delegates decoding/encoding to pluggable backends. Two are bundled:

| Name   | Package  | Notes                               |
| ------ | -------- | ----------------------------------- |
| opencv | OpenCV   | Default. Fast, ubiquitous.          |
| pyav   | PyAV/FFmpeg | Optional, install `pip install av` |

Select backend via the constructor:

```python
sv.Video("clip.mkv", backend="pyav")
```

## Writing frames manually

```python
src = sv.Video("input.mp4")
with src.sink("out.mp4") as sink:
    for fr in src.frames():
        sink.write(fr)
```

## Legacy helpers

Functions and classes in `supervision.utils.video` are **deprecated** and will be removed in 5 releases. They transparently wrap the new API and emit a `SupervisionWarnings` warning. Migrate as soon as convenient.
