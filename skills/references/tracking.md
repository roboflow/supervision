# Tracking (ByteTrack)

`sv.ByteTrack` assigns a persistent `tracker_id` to each detection across frames. It is stateful — create **one** instance and reuse it for every frame of a given stream/video; creating a new instance per frame resets tracking.

## Setup and usage

```python
import supervision as sv

tracker = sv.ByteTrack(
    track_activation_threshold=0.25,  # NOT `confidence_threshold`
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=30,
)

trace_annotator = sv.TraceAnnotator()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

for frame in frames:
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)

    detections = tracker.update_with_detections(detections)  # NOT `.update(detections)`

    labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

    annotated = trace_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(
        scene=annotated, detections=detections, labels=labels
    )
```

## Correct parameter names

`ByteTrack.__init__` uses:

- `track_activation_threshold` — minimum detection confidence to start a new track (this is the one most often mistyped as `confidence_threshold` or `track_thresh`).
- `lost_track_buffer` — number of frames to keep a track alive with no matching detection before dropping it.
- `minimum_matching_threshold` — IOU threshold for matching detections to existing tracks.
- `frame_rate` — expected FPS of the input, used for buffer timing.

## Correct method name

- `tracker.update_with_detections(detections) -> Detections` — the only public update method. It returns a **new** `Detections` object with `tracker_id` populated (and detections that couldn't be matched/confirmed may be dropped, so the returned length can be `<= len(detections)`).

```python
# WRONG — no such method
detections = tracker.update(detections)

# RIGHT
detections = tracker.update_with_detections(detections)
```

## Filtering by tracker_id

After tracking, `tracker_id` is just another attribute you can boolean-index on, same as `class_id`:

```python
# only detections that have been assigned a tracker id (drop unmatched, if any)
detections = detections[detections.tracker_id is not None]

# keep only a specific tracked object
detections = detections[detections.tracker_id == 7]

# exclude ids you've already counted/processed
seen_ids = {1, 2, 3}
detections = detections[~np.isin(detections.tracker_id, list(seen_ids))]
```

Note `tracker_id` is `None` on a `Detections` object until it has been passed through `tracker.update_with_detections(...)` at least once — accessing it before that raises/returns `None`, it is not auto-populated by `from_ultralytics` / `from_inference` alone.
