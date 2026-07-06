# Detections

`sv.Detections` is a single dataclass-like container used across the whole library. Every model integration converts its native output into one of these — always prefer the `from_*` constructor over building `Detections(...)` by hand.

## Creating Detections from common sources

```python
import supervision as sv

# Ultralytics (YOLOv8/v9/v10/11, SAM, etc.)
result = model(image)[0]
detections = sv.Detections.from_ultralytics(result)

# Roboflow inference (hosted or `inference` package)
result = model.infer(image)[0]
detections = sv.Detections.from_inference(result)

# Segment Anything (SAM / SAM2 / SAM3)
sam_result = mask_generator.generate(image)
detections = sv.Detections.from_sam(sam_result)

# Transformers (e.g. DETR, Grounding DINO via HF pipeline)
detections = sv.Detections.from_transformers(transformers_results, id2label=id2label)
```

Each `from_*` method normalizes the model's native output into the same set of attributes below — this is the whole point of using them instead of parsing raw model output yourself.

## Key attributes

| attribute    | shape / type                     | notes                                                                                                 |
| ------------ | -------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `xyxy`       | `np.ndarray (N, 4)`              | float, `[x1, y1, x2, y2]` per box, always present                                                     |
| `confidence` | `np.ndarray (N,)` or `None`      | float scores                                                                                          |
| `class_id`   | `np.ndarray (N,)` or `None`      | integer class ids                                                                                     |
| `tracker_id` | `np.ndarray (N,)` or `None`      | set after running a tracker, not by detection alone                                                   |
| `mask`       | `np.ndarray (N, H, W)` or `None` | boolean segmentation masks                                                                            |
| `data`       | `dict`                           | extra per-detection arrays, e.g. `data["class_name"]`; also accessible via `detections["class_name"]` |

`len(detections)` gives the number of boxes (`N`). `Detections` is empty-safe: with zero detections, arrays have shape `(0, 4)` / `(0,)` rather than being `None`.

## Filtering patterns

`Detections` supports NumPy-style boolean-mask indexing directly — this is the correct and idiomatic way to filter. It does **not** have a `.filter()` method.

```python
# keep only class_id == 0 (e.g. "person")
detections = detections[detections.class_id == 0]

# confidence threshold
detections = detections[detections.confidence > 0.5]

# combine conditions
detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5)]

# keep detections inside a set of classes
detections = detections[np.isin(detections.class_id, [0, 2, 3])]

# by area
detections = detections[detections.area > 1000]

# slicing / indexing a single detection
first = detections[0]
```

`Detections` also supports `+` to merge two instances and `sv.Detections.merge([d1, d2])` for combining more than two.

## Common mistakes

```python
# WRONG — Detections has no .filter() method
detections = detections.filter(lambda d: d.class_id == 0)

# RIGHT — boolean-mask indexing
detections = detections[detections.class_id == 0]
```

```python
# WRONG — comparing class_id to a class name string
detections = detections[detections.class_id == "person"]

# RIGHT — class_id is an integer id; compare to the class name via class_name data,
# or map the name to its integer id first
detections = detections[detections["class_name"] == "person"]
```

```python
# WRONG — assuming confidence/class_id are always populated
avg_conf = detections.confidence.mean()  # crashes if confidence is None

# RIGHT — guard when the source may not populate a field (e.g. some SAM masks
# have no class_id/confidence)
if detections.confidence is not None:
    avg_conf = detections.confidence.mean()
```
