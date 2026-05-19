---
comments: true
status: deprecated
---

# Deprecated

These features are phased out due to better alternatives or potential issues in future versions. Deprecated functionalities are typically supported for multiple subsequent releases, providing time for users to transition to updated methods.

- [`sv.ByteTrack`](https://supervision.roboflow.com/latest/trackers/#supervision.tracker.byte_tracker.core.ByteTrack) is deprecated in favour of `ByteTrackTracker` from the external [`trackers`](https://pypi.org/project/trackers/) package (`pip install trackers`). The update method is renamed from `update_with_detections()` to `update()`. Removal planned for `supervision-0.30.0`.
- `supervision.keypoint` module is deprecated; use `supervision.key_points` instead. Will be removed in `supervision-0.30.0`.
- `create_tiles` in `supervision.utils.image` is deprecated. Will be removed in `supervision-0.31.0`.
- `ensure_cv2_image_for_processing` in `supervision.utils.conversion` is deprecated. Will be removed in `supervision-0.31.0`.
- Keypoint validation utilities in `supervision.validators` are deprecated. Will be removed in `supervision-0.31.0`.
- `normalized_xyxy` argument in [`sv.denormalize_boxes`](https://supervision.roboflow.com/latest/detection/utils/boxes/#supervision.detection.utils.boxes.denormalize_boxes) is renamed to `xyxy`. Passing `normalized_xyxy=` emits a `FutureWarning`; support will be removed in `supervision-0.30.0`.
- `supervision.dataset.utils` import path for [`sv.rle_to_mask`](https://supervision.roboflow.com/latest/detection/utils/converters/#supervision.detection.utils.converters.rle_to_mask) and [`sv.mask_to_rle`](https://supervision.roboflow.com/latest/detection/utils/converters/#supervision.detection.utils.converters.mask_to_rle) is deprecated. These functions moved to `supervision.detection.utils.converters`. Will be removed in `supervision-0.30.0`.
- `sv.LMM` enum is deprecated and will be removed in `supervision-0.31.0`. Use `sv.VLM` instead.
- [`sv.Detections.from_lmm`](https://supervision.roboflow.com/latest/detection/core/#supervision.detection.core.Detections.from_lmm) property is deprecated and will be removed in `supervision-0.31.0`. Use [`sv.Detections.from_vlm`](https://supervision.roboflow.com/latest/detection/core/#supervision.detection.core.Detections.from_vlm) instead.

# Removed

### 0.27.0

- `overlap_ratio_wh` parameter in [`sv.InferenceSlicer`](https://supervision.roboflow.com/latest/detection/tools/inference_slicer/) has been removed. Use the pixel-based `overlap_wh` parameter instead.
- `overlap_filter_strategy` parameter in [`sv.InferenceSlicer`](https://supervision.roboflow.com/latest/detection/tools/inference_slicer/) has been removed. Use `overlap_strategy` instead.

### 0.26.0

- The `sv.DetectionDataset.images` property has been removed in `supervision-0.26.0`. Please loop over images with `for path, image, annotation in dataset:`, as that does not require loading all images into memory. Also, constructing `sv.DetectionDataset` with parameter `images` as `Dict[str, np.ndarray]` is deprecated and has been removed in `supervision-0.26.0`. Please pass a list of paths `List[str]` instead.
- The name `sv.BoundingBoxAnnotator` is deprecated and has been removed in `supervision-0.26.0`. It has been renamed to [`sv.BoxAnnotator`](https://supervision.roboflow.com/0.22.0/detection/annotators/#supervision.annotators.core.BoxAnnotator).

### 0.24.0

- The `frame_resolution_wh ` parameter in [`sv.PolygonZone`](detection/tools/polygon_zone.md/#supervision.detection.tools.polygon_zone.PolygonZone) has been removed.
- Supervision installation methods `"headless"` and `"desktop"` were removed, as they are no longer needed. `pip install supervision[headless]` will install the base library and harmlessly warn of non-existent extras.

### 0.23.0

- The `track_buffer`, `track_thresh`, and `match_thresh` parameters in [`ByteTrack`](trackers.md/#supervision.tracker.byte_tracker.core.ByteTrack) are deprecated and were removed as of `supervision-0.23.0`. Use `lost_track_buffer,` `track_activation_threshold`, and `minimum_matching_threshold` instead.
- The `triggering_position ` parameter in [`sv.PolygonZone`](detection/tools/polygon_zone.md/#supervision.detection.tools.polygon_zone.PolygonZone) was removed as of `supervision-0.23.0`. Use `triggering_anchors` instead.

### 0.22.0

- `sv.Detections.from_roboflow` is removed as of `supervision-0.22.0`. Use [`Detections.from_inference`](detection/core.md/#supervision.detection.core.Detections.from_inference) instead.
- The method `sv.Color.white()` was removed as of `supervision-0.22.0`. Use the constant `sv.Color.WHITE` instead.
- The method `sv.Color.black()` was removed as of `supervision-0.22.0`. Use the constant `sv.Color.BLACK` instead.
- The method `sv.Color.red()` was removed as of `supervision-0.22.0`. Use the constant `sv.Color.RED` instead.
- The method `sv.Color.green()` was removed as of `supervision-0.22.0`. Use the constant `sv.Color.GREEN` instead.
- The method `sv.Color.blue()` was removed as of `supervision-0.22.0`. Use the constant `sv.Color.BLUE` instead.
- The method `sv.ColorPalette.default()` was removed as of `supervision-0.22.0`. Use the constant [`ColorPalette.DEFAULT`](utils/draw.md/#supervision.draw.color.ColorPalette.DEFAULT) instead.
- `sv.BoxAnnotator` was removed as of `supervision-0.22.0`, however `sv.BoundingBoxAnnotator` was immediately renamed to `sv.BoxAnnotator`. Use [`BoxAnnotator`](detection/annotators.md/#supervision.annotators.core.BoxAnnotator) and [`LabelAnnotator`](detection/annotators.md/#supervision.annotators.core.LabelAnnotator) instead of the old `sv.BoxAnnotator`.
- The method `sv.FPSMonitor.__call__` was removed as of `supervision-0.22.0`. Use the attribute [`sv.FPSMonitor.fps`](utils/video.md/#supervision.utils.video.FPSMonitor.fps) instead.
