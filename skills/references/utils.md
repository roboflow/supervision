# Zones, colors, and other utils

## PolygonZone

`sv.PolygonZone` tests which detections fall inside an arbitrary polygon region
(e.g. a parking spot, a doorway, a lane). It does not draw anything itself — pair
it with `sv.PolygonZoneAnnotator` for visualization.

```python
import numpy as np
import supervision as sv

polygon = np.array([[100, 200], [400, 200], [400, 500], [100, 500]])

zone = sv.PolygonZone(polygon=polygon)
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED)

for frame in frames:
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)

    mask = zone.trigger(detections=detections)  # bool array, one per detection
    detections_in_zone = detections[mask]

    annotated = zone_annotator.annotate(scene=frame.copy())
    annotated = box_annotator.annotate(scene=annotated, detections=detections_in_zone)
```

`zone.trigger(detections)` returns a boolean NumPy array the same length as
`detections` — `True` where that detection's anchor point is inside the polygon.
Use it directly as a boolean mask; it does not filter `detections` for you.
`zone.current_count` holds the count of detections inside the zone after the most
recent `trigger()` call.

## LineZone for counting

`sv.LineZone` counts detections crossing a line, split into two directions.

```python
start, end = sv.Point(0, 300), sv.Point(1280, 300)
line_zone = sv.LineZone(start=start, end=end)
line_zone_annotator = sv.LineZoneAnnotator()

for frame in frames:
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = tracker.update_with_detections(detections)  # LineZone needs tracker_id

    line_zone.trigger(detections=detections)

    annotated = line_zone_annotator.annotate(frame=frame.copy(), line_counter=line_zone)

print(line_zone.in_count, line_zone.out_count)
```

- `line_zone.in_count` / `line_zone.out_count` — cumulative counters, updated
  in-place by each `trigger()` call (there's no return value to capture).
- `LineZone.trigger` requires `detections.tracker_id` to be populated — run a
  tracker (`sv.ByteTrack`) before calling it, otherwise a crossing can't be
  attributed to a consistent object between frames.

## sv.Color constants and from_hex

```python
sv.Color.RED
sv.Color.GREEN
sv.Color.BLUE
sv.Color.BLACK
sv.Color.WHITE

sv.Color.from_hex("#FF5733")
sv.Color.from_hex("FF5733")   # leading # optional
sv.Color.from_rgb_tuple((255, 87, 51))
```

## sv.ColorPalette.DEFAULT

Most annotators default to `sv.ColorPalette.DEFAULT` when no `color=` is given,
which cycles a distinct color per class/track index automatically — you usually
don't need to build a custom palette unless you want specific brand colors.

```python
box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.DEFAULT)

# custom palette
palette = sv.ColorPalette.from_hex(["#e6194b", "#3cb44b", "#ffe119", "#4363d8"])
box_annotator = sv.BoxAnnotator(color=palette, color_lookup=sv.ColorLookup.CLASS)
```
