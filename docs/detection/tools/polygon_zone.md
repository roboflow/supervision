---
comments: true
---

<div class="md-typeset">
  <h2>PolygonZone</h2>
</div>

:::supervision.detection.tools.polygon_zone.PolygonZone

## Occupancy

`PolygonZone.get_occupancy` estimates how much of a zone is covered by detections. It returns a value from `0.0` to `1.0`, where `0.0` means no detection pixels overlap the zone and `1.0` means the zone is fully occupied.

```python
import numpy as np
import supervision as sv

polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
zone = sv.PolygonZone(polygon=polygon)
detections = sv.Detections(xyxy=np.array([[0, 0, 50, 100]], dtype=np.float32))

occupancy = zone.get_occupancy(detections)
```

When detections include segmentation masks, occupancy is calculated from the mask pixels. Otherwise, detection boxes are rasterized. Overlapping detections are counted once.

<div class="md-typeset">
  <h2>PolygonZoneAnnotator</h2>
</div>

:::supervision.detection.tools.polygon_zone.PolygonZoneAnnotator
