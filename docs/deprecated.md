---
comments: true
status: deprecated
---

These features are phased out due to better alternatives or potential issues in future versions. Deprecated functionalities are supported for **three subsequent releases**, providing time for users to transition to updated methods.

- The `track_buffer`, `track_thresh`, and `match_thresh` parameters in [`ByterTrack`](trackers.md/#supervision.tracker.byte_tracker.core.ByteTrack) are deprecated and will be removed in `supervision-0.23.0`. Use `lost_track_buffer,` `track_activation_threshold`, and `minimum_matching_threshold` instead.
- The `triggering_position ` parameter in [`sv.PolygonZone`](detection/tools/polygon_zone.md/#supervision.detection.tools.polygon_zone.PolygonZone) is deprecated and will be removed in `supervision-0.23.0`. Use `triggering_anchors ` instead.
- The `frame_resolution_wh ` parameter in [`sv.PolygonZone`](detection/tools/polygon_zone.md/#supervision.detection.tools.polygon_zone.PolygonZone) is deprecated and will be removed in `supervision-0.24.0`.
- Constructing `DetectionDataset` with parameter `images` as `Dict[str, np.ndarray]` is deprecated and will be removed in `supervision-0.26.0`. Please pass a list of paths `List[str]` instead.
- The `DetectionDataset.images` property is deprecated and will be removed in `supervision-0.26.0`. Please loop over images with `for path, image, annotation in dataset:`, as that does not require loading all images into memory.
