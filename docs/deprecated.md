---
comments: true
status: deprecated
---

# Deprecated

These features are phased out due to better alternatives or potential issues in future versions. Deprecated functionalities are supported for **five subsequent releases**, providing time for users to transition to updated methods.


- `overlap_filter_strategy` in [`InferenceSlicer.__init__`](https://supervision.roboflow.com/latest/detection/tools/inference_slicer/) is deprecated and will be removed in `supervision-0.27.0`. Use `overlap_strategy` instead.
- `overlap_ratio_wh` in [`InferenceSlicer.__init__`](https://supervision.roboflow.com/latest/detection/tools/inference_slicer/) is deprecated and will be removed in `supervision-0.27.0`. Use `overlap_wh` instead.
- `sv.LMM` enum is deprecated and will be removed in `supervision-0.31.0`. Use `sv.VLM` instead.
- [`sv.Detections.from_lmm`](https://supervision.roboflow.com/0.26.0/detection/core/#supervision.detection.core.Detections.from_lmm) property is deprecated and will be removed in `supervision-0.31.0`. Use [`sv.Detections.from_vlm`](https://supervision.roboflow.com/0.26.0/detection/core/#supervision.detection.core.Detections.from_vlm) instead.
- [`sv.VideoInfo`](https://supervision.roboflow.com/latest/utils/video/#supervision.utils.video.VideoInfo) class is deprecated and will be removed in `supervision-0.32.0`. Use the new [`sv.Video`](https://supervision.roboflow.com/latest/utils/video/#supervision.utils.video_new.Video) API instead.
- [`sv.VideoSink`](https://supervision.roboflow.com/latest/utils/video/#supervision.utils.video.VideoSink) class is deprecated and will be removed in `supervision-0.32.0`. Use [`sv.Video.sink()`](https://supervision.roboflow.com/latest/utils/video/#supervision.utils.video_new.Video.sink) or [`sv.Video.save()`](https://supervision.roboflow.com/latest/utils/video/#supervision.utils.video_new.Video.save) instead.
- [`sv.get_video_frames_generator`](https://supervision.roboflow.com/latest/utils/video/#supervision.utils.video.get_video_frames_generator) function is deprecated and will be removed in `supervision-0.32.0`. Use [`sv.Video.frames()`](https://supervision.roboflow.com/latest/utils/video/#supervision.utils.video_new.Video.frames) instead.
- [`sv.process_video`](https://supervision.roboflow.com/latest/utils/video/#supervision.utils.video.process_video) function is deprecated and will be removed in `supervision-0.32.0`. Use [`sv.Video.save()`](https://supervision.roboflow.com/latest/utils/video/#supervision.utils.video_new.Video.save) instead.

# Removed

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
- The method `sv.ColorPalette.default()` was removed as of `supervision-0.22.0`. Use the constant [`ColorPalette.DEFAULT`](/utils/draw/#supervision.draw.color.ColorPalette.DEFAULT) instead.
- `sv.BoxAnnotator` was removed as of `supervision-0.22.0`, however `sv.BoundingBoxAnnotator` was immediately renamed to `sv.BoxAnnotator`. Use [`BoxAnnotator`](detection/annotators.md/#supervision.annotators.core.BoxAnnotator) and [`LabelAnnotator`](detection/annotators.md/#supervision.annotators.core.LabelAnnotator) instead of the old `sv.BoxAnnotator`.
- The method `sv.FPSMonitor.__call__` was removed as of `supervision-0.22.0`. Use the attribute [`sv.FPSMonitor.fps`](utils/video.md/#supervision.utils.video.FPSMonitor.fps) instead.
