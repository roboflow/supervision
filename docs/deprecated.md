---
comments: true
status: deprecated
---

These features are phased out due to better alternatives or potential issues in future versions. Deprecated functionalities are supported for **three subsequent releases**, providing time for users to transition to updated methods.

- [`Detections.from_froboflow`](detection/core.md/#supervision.detection.core.Detections.from_roboflow) is deprecated and will be removed in `supervision-0.22.0`. Use [`Detections.from_inference`](detection/core.md/#supervision.detection.core.Detections.from_inference) instead.
- `Color.white()` is deprecated and will be removed in `supervision-0.22.0`. Use `Color.WHITE` instead.
- `Color.black()` is deprecated and will be removed in `supervision-0.22.0`. Use `Color.BLACK` instead.
- `Color.red()` is deprecated and will be removed in `supervision-0.22.0`. Use `Color.RED` instead.
- `Color.green()` is deprecated and will be removed in `supervision-0.22.0`. Use `Color.GREEN` instead.
- `Color.blue()` is deprecated and will be removed in `supervision-0.22.0`. Use `Color.BLUE` instead.
- [`ColorPalette.default()`](draw/color.md/#supervision.draw.color.ColorPalette.default) is deprecated and will be removed in `supervision-0.22.0`. Use [`ColorPalette.DEFAULT`](draw/color.md/#supervision.draw.color.ColorPalette.DEFAULT) instead.
- `BoxAnnotator` is deprecated and will be removed in `supervision-0.22.0`. Use [`BoundingBoxAnnotator`](annotators.md/#supervision.annotators.core.BoundingBoxAnnotator) and [`LabelAnnotator`](annotators.md/#supervision.annotators.core.LabelAnnotator) instead.
- [`FPSMonitor.__call__`](utils/video.md/#supervision.utils.video.FPSMonitor.__call__) is deprecated and will be removed in `supervision-0.22.0`. Use [`FPSMonitor.fps`](utils/video.md/#supervision.utils.video.FPSMonitor.fps) instead.
