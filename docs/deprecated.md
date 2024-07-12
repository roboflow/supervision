---
comments: true
status: deprecated
---

# Deprecated

These features are phased out due to better alternatives or potential issues in future versions. Deprecated functionalities are supported for **five subsequent releases**, providing time for users to transition to updated methods.

- The `track_buffer`, `track_thresh`, and `match_thresh` parameters in [`ByterTrack`](trackers.md/#supervision.tracker.byte_tracker.core.ByteTrack) are deprecated and will be removed in `supervision-0.23.0`. Use `lost_track_buffer,` `track_activation_threshold`, and `minimum_matching_threshold` instead.
- The `triggering_position ` parameter in [`sv.PolygonZone`](detection/tools/polygon_zone.md/#supervision.detection.tools.polygon_zone.PolygonZone) will be removed in `supervision-0.23.0`. Use `triggering_anchors ` instead.
- The `frame_resolution_wh ` parameter in [`sv.PolygonZone`](detection/tools/polygon_zone.md/#supervision.detection.tools.polygon_zone.PolygonZone) will be removed in `supervision-0.24.0`.
- Constructing `DetectionDataset` and `ClassificationDataset` with parameter `images` as `Dict[str, np.ndarray]` will be removed in `supervision-0.26.0`. Please pass a list of paths `List[str]` instead.
- The `DetectionDataset.images` property will be removed in `supervision-0.26.0`. Please loop over images with `for path, image, annotation in dataset:`, as that does not require loading all images into memory.
- `BoundingBoxAnnotator` has been renamed to `BoxAnnotator` after the old implementation of `BoxAnnotator` has been removed. `BoundingBoxAnnotator` will be removed in `supervision-0.26.0`.

# Removed

- [`Detections.from_froboflow`](detection/core.md/#supervision.detection.core.Detections.from_roboflow) is removed as of `supervision-0.22.0`. Use [`Detections.from_inference`](detection/core.md/#supervision.detection.core.Detections.from_inference) instead.
- The method `Color.white()` was removed as of `supervision-0.22.0`. Use the constant `Color.WHITE` instead.
- The method `Color.black()` was removed as of `supervision-0.22.0`. Use the constant `Color.BLACK` instead.
- The method `Color.red()` was removed as of `supervision-0.22.0`. Use the constant `Color.RED` instead.
- The method `Color.green()` was removed as of `supervision-0.22.0`. Use the constant `Color.GREEN` instead.
- The method `Color.blue()` was removed as of `supervision-0.22.0`. Use the constant `Color.BLUE` instead.
- The method [`ColorPalette.default()`](draw/color.md/#supervision.draw.color.ColorPalette.default) was removed as of `supervision-0.22.0`. Use the constant [`ColorPalette.DEFAULT`](draw/color.md/#supervision.draw.color.ColorPalette.DEFAULT) instead.
- `BoxAnnotator` was removed as of `supervision-0.22.0`, however `BoundingBoxAnnotator` was immediately renamed to `BoxAnnotator`. Use [`BoxAnnotator`](detection/annotators.md/#supervision.annotators.core.BoxAnnotator) and [`LabelAnnotator`](detection/annotators.md/#supervision.annotators.core.LabelAnnotator) instead of the old `BoxAnnotator`.
- The method [`FPSMonitor.__call__`](utils/video.md/#supervision.utils.video.FPSMonitor.__call__) was removed as of `supervision-0.22.0`. Use the attribute [`FPSMonitor.fps`](utils/video.md/#supervision.utils.video.FPSMonitor.fps) instead.
