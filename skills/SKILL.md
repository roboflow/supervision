---
name: supervision
description: use when working with roboflow/supervision
---

# supervision

`supervision` is a model-agnostic computer vision library (roboflow/supervision) for
working with detection/segmentation results: building `sv.Detections`, drawing with
annotators, tracking objects across frames, and processing video/streams.

This skill covers the patterns an agent gets wrong most often. For anything not
covered here, read the source under `src/supervision/` rather than guessing at an
API — many method/parameter names look plausible but don't exist (see "common
mistakes" in each reference file).

## Most common pattern: detect + annotate

The standard loop is: run a model, wrap its output in `sv.Detections`, draw boxes
and labels, output/save the frame.

```python
import cv2
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

image = cv2.imread("image.jpg")
result = model(image)[0]
detections = sv.Detections.from_ultralytics(result)

labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(detections["class_name"], detections.confidence)
]

annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

cv2.imwrite("annotated.jpg", annotated)
```

See `references/detection.md` for building `Detections` from other sources
(inference, SAM3) and filtering, and `references/annotators.md` for the full
annotator list and the compose pattern.

The repo also publishes `docs/llms.txt` for general model-level facts and API surface; this skill focuses specifically on the mistakes agents repeatedly make in practice (wrong method names, deprecated APIs, silently-ignored kwargs) with runnable patterns, rather than restating the API reference.

## Critical decision: InferencePipeline vs sv.process_video

These solve the same problem — "run a model over every frame of a video/stream" —
but they are not interchangeable. Picking the wrong one is the single most common
architectural mistake in supervision code.

**Use `sv.process_video` when:**
- The input is a finite video *file* on disk and you want an output video file.
- You bring your own model call inside a `callback(frame, frame_index) -> np.ndarray`.
- You want the simplest possible script — no threading, no queues.

```python
import supervision as sv

def callback(frame: sv.numpy.ndarray, frame_index: int) -> sv.numpy.ndarray:
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    return box_annotator.annotate(scene=frame.copy(), detections=detections)

sv.process_video(
    source_path="input.mp4",
    target_path="output.mp4",
    callback=callback,
    show_progress=True,  # NOT `progress=True` — see references/video.md
)
```

**Use `roboflow/inference`'s `InferencePipeline` when:**
- The source is a *live* stream — webcam, RTSP, or an infinite feed — not a video
  file you're transcoding.
- You need the model inference itself decoupled/threaded from frame reading for
  real-time throughput (InferencePipeline runs inference in a background thread and
  calls your `on_prediction` callback as results become available).
- You don't need a saved output video, or you'll build one yourself (e.g. with
  `sv.VideoSink`) inside the callback.
- You want Roboflow-hosted or local Roboflow models run for you, rather than
  calling `model(frame)` yourself each iteration.

```python
from inference import InferencePipeline
import supervision as sv

box_annotator = sv.BoxAnnotator()

def on_prediction(result, video_frame):
    detections = sv.Detections.from_inference(result)
    annotated = box_annotator.annotate(scene=video_frame.image.copy(), detections=detections)
    cv2.imshow("frame", annotated)
    cv2.waitKey(1)

pipeline = InferencePipeline.init(
    model_id="your-model/1",
    video_reference=0,  # webcam, or an RTSP URL
    on_prediction=on_prediction,
)
pipeline.start()
pipeline.join()
```

**Rule of thumb:** file in, file out → `sv.process_video`. Live/streaming source,
or you need async/threaded inference → `InferencePipeline`. Don't reach for
`InferencePipeline` just to process a static mp4 — it adds threading complexity
`process_video` doesn't need, and don't use `process_video` on an infinite/live
source — it assumes a finite frame count from `VideoInfo`.

## Reference files

- `references/detection.md` — building `sv.Detections`, key attributes, filtering,
  common mistakes.
- `references/annotators.md` — annotator classes, correct parameter names, the
  compose pattern.
- `references/tracking.md` — tracking with the `trackers` package (`ByteTrackTracker`),
  why `sv.ByteTrack` is deprecated, correct parameter/method names, filtering by
  `tracker_id`.
- `references/video.md` — `sv.process_video`, `VideoInfo`, `VideoSink`.
- `references/utils.md` — `PolygonZone`, `LineZone`, `sv.Color` / `sv.ColorPalette`.
