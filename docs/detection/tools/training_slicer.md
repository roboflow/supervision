---
comments: true
---

# TrainingSlicer

`TrainingSlicer` is the training-time counterpart to [`InferenceSlicer`](inference_slicer.md). Instead of running a model callback on each tile, it slices an image's existing ground-truth `Detections` to match a grid of tiles, so you can turn a dataset of large images into fixed-size training crops — the standard technique for training small-object detectors.

```python
import numpy as np
import supervision as sv

image = np.zeros((1024, 1024, 3), dtype=np.uint8)
detections = sv.Detections(
    xyxy=np.array([[100, 100, 180, 180], [900, 900, 980, 980]], dtype=float),
    class_id=np.array([0, 1]),
)

slicer = sv.TrainingSlicer(slice_wh=320, overlap_wh=0)
tiles = slicer(image, detections)

for tile_image, tile_detections in tiles:
    print(tile_image.shape, len(tile_detections))
```

Annotations that are cut by a tile boundary are clipped to that tile, and dropped from a tile entirely once too little of the original box remains visible there — controlled by `min_visibility`:

```python
slicer = sv.TrainingSlicer(slice_wh=320, overlap_wh=0, min_visibility=0.3)
```

By default, tiles with no annotations are still returned (useful for background/hard-negative training samples). Pass `drop_empty_slices=True` to keep only tiles that contain at least one annotation:

```python
slicer = sv.TrainingSlicer(slice_wh=320, overlap_wh=0, drop_empty_slices=True)
```

:::supervision.detection.tools.training_slicer.TrainingSlicer
