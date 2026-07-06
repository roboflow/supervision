# Tracking

## sv.ByteTrack is deprecated — use the `trackers` package

`supervision`'s built-in `sv.ByteTrack` is deprecated. The current, maintained way to track objects is the standalone `trackers` package (`pip install trackers`), which provides `ByteTrackTracker` (plus `SORTTracker`, `OCSORTTracker`, `BoTSORTTracker`). It still consumes/returns `sv.Detections`, so everything else in this skill (filtering, annotating) works unchanged — only the tracker object and its update method differ.

```python
import cv2
import supervision as sv
from trackers import ByteTrackTracker

tracker = ByteTrackTracker(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_consecutive_frames=3,
    minimum_iou_threshold=0.3,
)

trace_annotator = sv.TraceAnnotator()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

for frame in frames:
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)

    detections = tracker.update(detections)  # NOT `.update_with_detections()`

    labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

    annotated = trace_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

tracker.reset()  # call between videos/streams to clear track state
```

`ByteTrackTracker` is stateful, like the old `sv.ByteTrack` — create **one** instance per video/stream and reuse it every frame; creating a new instance per frame resets tracking.

### Correct constructor parameters (`ByteTrackTracker`)

- `track_activation_threshold` (default `0.25`) — minimum detection confidence to start a new track. Not `confidence_threshold`.
- `lost_track_buffer` (default `30`) — frames to keep a track alive with no matching detection before dropping it.
- `minimum_consecutive_frames` (default `3`) — consecutive detections required before a track is confirmed; suppresses spurious one-frame detections.
- `minimum_iou_threshold` (default `0.3`) — IOU threshold for matching detections to existing tracks. This replaced the old `minimum_matching_threshold` name.

### Correct method name

- `tracker.update(detections) -> Detections` — NOT `update_with_detections()` (that was the `sv.ByteTrack` method name) and NOT plain `update()` semantics from other libraries — pass the `Detections` object, get a new `Detections` back with `tracker_id` populated.
- `tracker.update(detections, timestamp=...)` — pass a monotonic `timestamp` in seconds if your pipeline has irregular/dropped frames, so Kalman prediction and lost-track pruning match the real time gap instead of assuming a fixed frame rate.
- `tracker.reset()` — clears all track state; call this between videos, not just once at startup.

```python
# WRONG — this is the deprecated sv.ByteTrack method name
detections = tracker.update_with_detections(detections)

# RIGHT — ByteTrackTracker from the `trackers` package
detections = tracker.update(detections)
```

## Legacy: sv.ByteTrack (deprecated, still present in supervision)

You may still encounter `sv.ByteTrack` in older code. It behaves the same way conceptually but with different names — recognize it, don't write new code against it:

```python
tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,  # note: different default/name than the new package
    frame_rate=30,
)
detections = tracker.update_with_detections(detections)  # old method name
```

If you see this pattern in an existing codebase, prefer migrating it to `ByteTrackTracker` from `trackers` rather than extending it further.

## Filtering by tracker_id

After tracking, `tracker_id` is just another attribute you can boolean-index on, same as `class_id`:

```python
# only detections that have been assigned a tracker id (drop unmatched, if any)
detections = detections[detections.tracker_id != None]  # noqa: E711 (elementwise, not `is not None`)

# keep only a specific tracked object
detections = detections[detections.tracker_id == 7]

# exclude ids you've already counted/processed
seen_ids = {1, 2, 3}
detections = detections[~np.isin(detections.tracker_id, list(seen_ids))]
```

Note `tracker_id` is `None` on a `Detections` object until it has been passed through `tracker.update(...)` at least once — accessing it before that raises/returns `None`, it is not auto-populated by `from_ultralytics` / `from_inference` alone. Also note `detections.tracker_id != None` (elementwise numpy comparison) is intentional here, not a mistake — `is not None` would do a Python identity check on the whole array instead of an elementwise mask.
