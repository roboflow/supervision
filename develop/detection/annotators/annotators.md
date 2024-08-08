---
comments: true
status: new
---

# Annotators

=== "Box"

    ```python
    import supervision as sv

    image = ...
    detections = sv.Detections(...)

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )
    ```

    <div class="result" markdown>

    ![bounding-box-annotator-example](https://media.roboflow.com/supervision-annotator-examples/bounding-box-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "RoundBox"

    ```python
    import supervision as sv

    image = ...
    detections = sv.Detections(...)

    round_box_annotator = sv.RoundBoxAnnotator()
    annotated_frame = round_box_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )
    ```

    <div class="result" markdown>

    ![round-box-annotator-example](https://media.roboflow.com/supervision-annotator-examples/round-box-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "BoxCorner"

    ```python
    import supervision as sv

    image = ...
    detections = sv.Detections(...)

    corner_annotator = sv.BoxCornerAnnotator()
    annotated_frame = corner_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )
    ```

    <div class="result" markdown>

    ![box-corner-annotator-example](https://media.roboflow.com/supervision-annotator-examples/box-corner-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "Color"

    ```python
    import supervision as sv

    image = ...
    detections = sv.Detections(...)

    color_annotator = sv.ColorAnnotator()
    annotated_frame = color_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )
    ```

    <div class="result" markdown>

    ![box-mask-annotator-example](https://media.roboflow.com/supervision-annotator-examples/box-mask-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "Circle"

    ```python
    import supervision as sv

    image = ...
    detections = sv.Detections(...)

    circle_annotator = sv.CircleAnnotator()
    annotated_frame = circle_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )
    ```

    <div class="result" markdown>

    ![circle-annotator-example](https://media.roboflow.com/supervision-annotator-examples/circle-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "Dot"

    ```python
    import supervision as sv

    image = ...
    detections = sv.Detections(...)

    dot_annotator = sv.DotAnnotator()
    annotated_frame = dot_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )
    ```

    <div class="result" markdown>

    ![dot-annotator-example](https://media.roboflow.com/supervision-annotator-examples/dot-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "Triangle"

    ```python
    import supervision as sv

    image = ...
    detections = sv.Detections(...)

    triangle_annotator = sv.TriangleAnnotator()
    annotated_frame = triangle_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )
    ```

    <div class="result" markdown>

    ![triangle-annotator-example](https://media.roboflow.com/supervision-annotator-examples/triangle-annotator-example.png){ align=center width="800" }

    </div>

=== "Ellipse"

    ```python
    import supervision as sv

    image = ...
    detections = sv.Detections(...)

    ellipse_annotator = sv.EllipseAnnotator()
    annotated_frame = ellipse_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )
    ```

    <div class="result" markdown>

    ![ellipse-annotator-example](https://media.roboflow.com/supervision-annotator-examples/ellipse-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "Halo"

    ```python
    import supervision as sv

    image = ...
    detections = sv.Detections(...)

    halo_annotator = sv.HaloAnnotator()
    annotated_frame = halo_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )
    ```

    <div class="result" markdown>

    ![halo-annotator-example](https://media.roboflow.com/supervision-annotator-examples/halo-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "PercentageBar"

    ```python
    import supervision as sv

    image = ...
    detections = sv.Detections(...)

    percentage_bar_annotator = sv.PercentageBarAnnotator()
    annotated_frame = percentage_bar_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )
    ```

    <div class="result" markdown>

    ![percentage-bar-annotator-example](https://media.roboflow.com/supervision-annotator-examples/percentage-bar-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "Mask"

    ```python
    import supervision as sv

    image = ...
    detections = sv.Detections(...)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )
    ```

    <div class="result" markdown>

    ![mask-annotator-example](https://media.roboflow.com/supervision-annotator-examples/mask-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "Polygon"

    ```python
    import supervision as sv

    image = ...
    detections = sv.Detections(...)

    polygon_annotator = sv.PolygonAnnotator()
    annotated_frame = polygon_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )
    ```

    <div class="result" markdown>

    ![polygon-annotator-example](https://media.roboflow.com/supervision-annotator-examples/polygon-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "Label"

    ```python
    import supervision as sv

    image = ...
    detections = sv.Detections(...)

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(detections['class_name'], detections.confidence)
    ]

    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
    annotated_frame = label_annotator.annotate(
        scene=image.copy(),
        detections=detections,
        labels=labels
    )
    ```

    <div class="result" markdown>

    ![label-annotator-example](https://media.roboflow.com/supervision-annotator-examples/label-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "RichLabel"

    ```python
    import supervision as sv

    image = ...
    detections = sv.Detections(...)

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(detections['class_name'], detections.confidence)
    ]

    rich_label_annotator = sv.RichLabelAnnotator(
        font_path="<TTF_FONT_PATH>",
        text_position=sv.Position.CENTER
    )
    annotated_frame = rich_label_annotator.annotate(
        scene=image.copy(),
        detections=detections,
        labels=labels
    )
    ```

    <div class="result" markdown>

    ![label-annotator-example](https://media.roboflow.com/supervision-annotator-examples/label-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "Crop"

    ```python
    import supervision as sv

    image = ...
    detections = sv.Detections(...)

    crop_annotator = sv.CropAnnotator()
    annotated_frame = crop_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )
    ```

=== "Blur"

    ```python
    import supervision as sv

    image = ...
    detections = sv.Detections(...)

    blur_annotator = sv.BlurAnnotator()
    annotated_frame = blur_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )
    ```

    <div class="result" markdown>

    ![blur-annotator-example](https://media.roboflow.com/supervision-annotator-examples/blur-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "Pixelate"

    ```python
    import supervision as sv

    image = ...
    detections = sv.Detections(...)

    pixelate_annotator = sv.PixelateAnnotator()
    annotated_frame = pixelate_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )
    ```

    <div class="result" markdown>

    ![pixelate-annotator-example](https://media.roboflow.com/supervision-annotator-examples/pixelate-annotator-example-10.png){ align=center width="800" }

    </div>

=== "Trace"

    ```python
    import supervision as sv
    from ultralytics import YOLO

    model = YOLO('yolov8x.pt')

    trace_annotator = sv.TraceAnnotator()

    video_info = sv.VideoInfo.from_video_path(video_path='...')
    frames_generator = get_video_frames_generator(source_path='...')
    tracker = sv.ByteTrack()

    with sv.VideoSink(target_path='...', video_info=video_info) as sink:
        for frame in frames_generator:
            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)
            annotated_frame = trace_annotator.annotate(
                scene=frame.copy(),
                detections=detections)
            sink.write_frame(frame=annotated_frame)
    ```

    <div class="result" markdown>

    ![trace-annotator-example](https://media.roboflow.com/supervision-annotator-examples/trace-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "HeatMap"

    ```python
    import supervision as sv
    from ultralytics import YOLO

    model = YOLO('yolov8x.pt')

    heat_map_annotator = sv.HeatMapAnnotator()

    video_info = sv.VideoInfo.from_video_path(video_path='...')
    frames_generator = get_video_frames_generator(source_path='...')

    with sv.VideoSink(target_path='...', video_info=video_info) as sink:
        for frame in frames_generator:
            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            annotated_frame = heat_map_annotator.annotate(
                scene=frame.copy(),
                detections=detections)
            sink.write_frame(frame=annotated_frame)
    ```

    <div class="result" markdown>

    ![heat-map-annotator-example](https://media.roboflow.com/supervision-annotator-examples/heat-map-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "Background Color"

    ```python
    import supervision as sv

    image = ...
    detections = sv.Detections(...)

    background_overlay_annotator = sv.BackgroundOverlayAnnotator()
    annotated_frame = background_overlay_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )
    ```

    <div class="result" markdown>

    ![background-overlay-annotator-example](https://media.roboflow.com/supervision-annotator-examples/background-color-annotator-example-purple.png)

    </div>

<div class="md-typeset">
    <h2><a href="#supervision.annotators.core.BoxAnnotator">BoxAnnotator</a></h2>
</div>

:::supervision.annotators.core.BoxAnnotator

<div class="md-typeset">
    <h2><a href="#supervision.annotators.core.RoundBoxAnnotator">RoundBoxAnnotator</a></h2>
</div>

:::supervision.annotators.core.RoundBoxAnnotator

<div class="md-typeset">
    <h2><a href="#supervision.annotators.core.BoxCornerAnnotator">BoxCornerAnnotator</a></h2>
</div>

:::supervision.annotators.core.BoxCornerAnnotator

<div class="md-typeset">
    <h2><a href="#supervision.annotators.core.OrientedBoxAnnotator">OrientedBoxAnnotator</a></h2>
</div>

:::supervision.annotators.core.OrientedBoxAnnotator

<div class="md-typeset">
    <h2><a href="#supervision.annotators.core.ColorAnnotator">ColorAnnotator</a></h2>
</div>

:::supervision.annotators.core.ColorAnnotator

<div class="md-typeset">
    <h2><a href="#supervision.annotators.core.CircleAnnotator">CircleAnnotator</a></h2>
</div>

:::supervision.annotators.core.CircleAnnotator

<div class="md-typeset">
    <h2><a href="#supervision.annotators.core.DotAnnotator">DotAnnotator</a></h2>
</div>

:::supervision.annotators.core.DotAnnotator

<div class="md-typeset">
    <h2><a href="#supervision.annotators.core.TriangleAnnotator">TriangleAnnotator</a></h2>
</div>

:::supervision.annotators.core.TriangleAnnotator

<div class="md-typeset">
    <h2><a href="#supervision.annotators.core.EllipseAnnotator">EllipseAnnotator</a></h2>
</div>

:::supervision.annotators.core.EllipseAnnotator

<div class="md-typeset">
    <h2><a href="#supervision.annotators.core.HaloAnnotator">HaloAnnotator</a></h2>
</div>

:::supervision.annotators.core.HaloAnnotator

<div class="md-typeset">
    <h2><a href="#supervision.annotators.core.PercentageBarAnnotator">PercentageBarAnnotator</a></h2>
</div>

:::supervision.annotators.core.PercentageBarAnnotator

<div class="md-typeset">
    <h2><a href="#supervision.annotators.core.HeatMapAnnotator">HeatMapAnnotator</a></h2>
</div>

:::supervision.annotators.core.HeatMapAnnotator

<div class="md-typeset">
    <h2><a href="#supervision.annotators.core.MaskAnnotator">MaskAnnotator</a></h2>
</div>

:::supervision.annotators.core.MaskAnnotator

<div class="md-typeset">
    <h2><a href="#supervision.annotators.core.PolygonAnnotator">PolygonAnnotator</a></h2>
</div>

:::supervision.annotators.core.PolygonAnnotator

<div class="md-typeset">
    <h2><a href="#supervision.annotators.core.LabelAnnotator">LabelAnnotator</a></h2>
</div>

:::supervision.annotators.core.LabelAnnotator

<div class="md-typeset">
    <h2><a href="#supervision.annotators.core.RichLabelAnnotator">RichLabelAnnotator</a></h2>
</div>

:::supervision.annotators.core.RichLabelAnnotator

<div class="md-typeset">
    <h2><a href="#supervision.annotators.core.BlurAnnotator">BlurAnnotator</a></h2>
</div>

:::supervision.annotators.core.BlurAnnotator

<div class="md-typeset">
    <h2><a href="#supervision.annotators.core.PixelateAnnotator">PixelateAnnotator</a></h2>
</div>

:::supervision.annotators.core.PixelateAnnotator

<div class="md-typeset">
    <h2><a href="#supervision.annotators.core.TraceAnnotator">TraceAnnotator</a></h2>
</div>

:::supervision.annotators.core.TraceAnnotator

<div class="md-typeset">
    <h2><a href="#supervision.annotators.core.CropAnnotator">CropAnnotator</a></h2>
</div>

:::supervision.annotators.core.CropAnnotator

<div class="md-typeset">
    <h2><a href="#supervision.annotators.core.BackgroundOverlayAnnotator">BackgroundOverlayAnnotator</a></h2>
</div>

:::supervision.annotators.core.BackgroundOverlayAnnotator

<div class="md-typeset">
    <h2><a href="#supervision.annotators.core.ColorLookup">ColorLookup</a></h2>
</div>

:::supervision.annotators.utils.ColorLookup
