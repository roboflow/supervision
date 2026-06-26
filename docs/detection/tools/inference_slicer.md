---
comments: true
---

# InferenceSlicer

## GeoTIFF Datasets

Install the optional GeoTIFF dependencies before running this example:

```bash
pip install "supervision[geotiff]"
wget -O RGB.byte.tif https://raw.githubusercontent.com/rasterio/rasterio/main/tests/data/RGB.byte.tif
```

`InferenceSlicer` can read an open `rasterio` dataset window-by-window. This keeps large GeoTIFFs out of memory while passing each tile to the callback as an `(H, W, C)` NumPy array.

```python
import numpy as np
import rasterio
import supervision as sv


def callback(tile: np.ndarray) -> sv.Detections:
    h, w = tile.shape[:2]
    return sv.Detections(
        xyxy=np.array([[w * 0.25, h * 0.25, w * 0.75, h * 0.75]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )


slicer = sv.InferenceSlicer(
    callback=callback,
    slice_wh=(256, 256),
    overlap_wh=(64, 64),
    overlap_filter=sv.OverlapFilter.NONE,
)

with rasterio.open("RGB.byte.tif") as dataset:
    detections = slicer(dataset)

print(len(detections))
```

GeoTIFF inputs must use a projected coordinate reference system. Reproject geographic rasters before passing them to `InferenceSlicer`.

:::supervision.detection.tools.inference_slicer.InferenceSlicer
