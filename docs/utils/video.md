---
comments: true
---

# Video Utils

!!! tip "New Video API"

    A new, more powerful Video API is now available! The `Video` class provides a unified interface for video processing with support for files, streams, and webcams, along with multi-backend support (OpenCV and PyAV).

    ```python
    import supervision as sv

    # Simple usage
    video = sv.Video("source.mp4")
    for frame in video:
        # Process frame
        pass

    # Advanced features
    video.save("output.mp4", callback=process_frame, fps=60)
    ```

## New Video API

<div class="md-typeset">
    <h2><a href="#supervision.utils.video_new.Video">Video</a></h2>
</div>

:::supervision.utils.video_new.Video

<div class="md-typeset">
    <h2><a href="#supervision.utils.video_backend.VideoInfo">VideoInfo (New)</a></h2>
</div>

:::supervision.utils.video_backend.VideoInfo

## Legacy Video API (Deprecated)

!!! warning "Deprecation Notice"

    The following classes and functions are deprecated and will be removed in `supervision-0.32.0`. Please migrate to the new `Video` API above.

<div class="md-typeset">
    <h2><a href="#supervision.utils.video.VideoInfo">VideoInfo (Deprecated)</a></h2>
</div>

:::supervision.utils.video.VideoInfo

<div class="md-typeset">
    <h2><a href="#supervision.utils.video.VideoSink">VideoSink</a></h2>
</div>

:::supervision.utils.video.VideoSink

<div class="md-typeset">
    <h2><a href="#supervision.utils.video.FPSMonitor">FPSMonitor</a></h2>
</div>

:::supervision.utils.video.FPSMonitor

<div class="md-typeset">
    <h2><a href="#supervision.utils.video.get_video_frames_generator">get_video_frames_generator</a></h2>
</div>

:::supervision.utils.video.get_video_frames_generator

<div class="md-typeset">
    <h2><a href="#supervision.utils.video.process_video">process_video</a></h2>
</div>

:::supervision.utils.video.process_video
