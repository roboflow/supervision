# Annotators

All annotators implement `.annotate(scene, detections, ...) -> np.ndarray`. They never mutate `scene` in place as documented usage — pass `scene.copy()` (or reuse the returned array as the next annotator's input) rather than the original frame if you need the original preserved.

## Common annotator classes

| class                       | draws                                                    | notable params                                                       |
| --------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------- |
| `sv.BoxAnnotator`           | rectangle bounding boxes                                 | `color`, `thickness`                                                 |
| `sv.RoundBoxAnnotator`      | rounded-corner boxes                                     | `color`, `thickness`, `roundness`                                    |
| `sv.BoxCornerAnnotator`     | corner-only brackets                                     | `color`, `thickness`, `corner_length`                                |
| `sv.LabelAnnotator`         | text labels (needs `labels=[...]`)                       | `color`, `text_color`, `text_scale`, `text_padding`, `text_position` |
| `sv.RichLabelAnnotator`     | text labels with custom font/unicode support             | `font_path`, `text_color`, `text_scale`                              |
| `sv.MaskAnnotator`          | filled segmentation masks                                | `color`, `opacity`                                                   |
| `sv.PolygonAnnotator`       | mask/box outline as polygon                              | `color`, `thickness`                                                 |
| `sv.EllipseAnnotator`       | ellipse under each box (good for people-tracking)        | `color`, `thickness`, `start_angle`, `end_angle`                     |
| `sv.CircleAnnotator`        | circle around each box center                            | `color`, `thickness`                                                 |
| `sv.DotAnnotator`           | filled dot at box center/anchor                          | `color`, `radius`, `position`                                        |
| `sv.TriangleAnnotator`      | triangle marker above box                                | `color`, `base`, `height`                                            |
| `sv.HaloAnnotator`          | glow/halo around mask                                    | `color`, `opacity`, `kernel_size`                                    |
| `sv.HeatMapAnnotator`       | cumulative heatmap across frames                         | `position`, `opacity`, `radius`                                      |
| `sv.BlurAnnotator`          | blur out detected regions                                | `kernel_size`                                                        |
| `sv.PixelateAnnotator`      | pixelate detected regions                                | `pixel_size`                                                         |
| `sv.TraceAnnotator`         | draws tracked path history, needs `tracker_id`           | `color`, `position`, `trace_length`                                  |
| `sv.CropAnnotator`          | pastes a zoomed crop of each detection back on the scene | `position`, `scale`                                                  |
| `sv.IconAnnotator`          | places a custom image/icon at each detection             | `icon_path` or `icon_resolver`, `icon_scale`                         |
| `sv.PercentageBarAnnotator` | draws a confidence bar under each box                    | `color`, `height`, `width`                                           |

Most annotators accept `color=sv.Color(...)` / `sv.ColorPalette(...)` and a `color_lookup` argument (`sv.ColorLookup.CLASS`, `.INDEX`, `.TRACK`) controlling whether color is assigned per class, per detection index, or per tracker id.

## Compose pattern (chain annotators)

Annotators are meant to be chained — call each one with the previous output as the new `scene`:

```python
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

annotated = scene.copy()
annotated = trace_annotator.annotate(scene=annotated, detections=detections)
annotated = box_annotator.annotate(scene=annotated, detections=detections)
annotated = label_annotator.annotate(
    scene=annotated, detections=detections, labels=labels
)
```

Order matters visually — draw fills/masks/traces first, then outlines, then labels on top so text isn't obscured.

## Common mistake

```python
# WRONG — this class does not exist in supervision
annotator = sv.BoundingBoxAnnotator()

# RIGHT
annotator = sv.BoxAnnotator()
```

Other name mix-ups worth double-checking against `src/supervision/annotators/core.py` before using: `sv.LabelAnnotator` (not `TextAnnotator`), `sv.MaskAnnotator` (not `SegmentationAnnotator`), and `sv.TraceAnnotator` (not `PathAnnotator` / `TrajectoryAnnotator`).
