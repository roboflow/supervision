# Video

## sv.process_video

`sv.process_video` reads a finite video file frame-by-frame, applies a callback, and writes the result to a new video file. It handles the read/write loop and progress bar for you.

```python
import supervision as sv


def callback(frame, frame_index: int):
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    return box_annotator.annotate(scene=frame.copy(), detections=detections)


sv.process_video(
    source_path="input.mp4",
    target_path="output.mp4",
    callback=callback,
    show_progress=True,
)
```

### Critical: the parameter is `show_progress`, not `progress`

```python
# WRONG — `progress` is not a recognized kwarg; this silently does nothing
# (no error is raised, you just get no progress bar — easy to miss in code review)
sv.process_video(
    source_path="input.mp4", target_path="output.mp4", callback=callback, progress=True
)

# RIGHT
sv.process_video(
    source_path="input.mp4",
    target_path="output.mp4",
    callback=callback,
    show_progress=True,
)
```

Because passing an unexpected keyword like `progress` doesn't always raise immediately depending on the function signature/version, always verify the actual parameter name against the installed version's signature (`help(sv.process_video)`) rather than assuming.

## VideoInfo

`sv.VideoInfo.from_video_path(path)` returns metadata about a video file:

```python
video_info = sv.VideoInfo.from_video_path("input.mp4")

video_info.width  # int, pixels
video_info.height  # int, pixels
video_info.fps  # float — NOT guaranteed to be a whole number (e.g. 29.97)
video_info.total_frames  # int, or None if it can't be determined
```

`fps` is a **float**. Don't do `int(video_info.fps)` when the real value matters (e.g. computing timestamps from frame index) — truncating 29.97 to 29 introduces drift over a long video. Use it directly in float math:

```python
timestamp_seconds = frame_index / video_info.fps
```

## VideoSink for manual frame writing

Use `sv.VideoSink` directly (instead of `process_video`) when you need more control than a single callback gives you — e.g. skipping frames, writing frames from a live source, or writing inside an `InferencePipeline` callback.

```python
video_info = sv.VideoInfo.from_video_path("input.mp4")

with sv.VideoSink(target_path="output.mp4", video_info=video_info) as sink:
    for frame in sv.get_video_frames_generator(source_path="input.mp4"):
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
        sink.write_frame(annotated)
```

`sv.get_video_frames_generator(source_path=...)` is the underlying frame-reading generator `process_video` uses internally — reach for it when you need a `for` loop over frames instead of a callback style.
