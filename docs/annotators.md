=== "BoundingBox"

    ```python
    >>> import supervision as sv

    >>> image = ...
    >>> detections = sv.Detections(...)

    >>> bounding_box_annotator = sv.BoundingBoxAnnotator()
    >>> annotated_frame = bounding_box_annotator.annotate(
    ...     scene=image.copy(),
    ...     detections=detections
    ... )
    ```

    <div class="result" markdown>

    ![bounding-box-annotator-example](https://media.roboflow.com/supervision-annotator-examples/bounding-box-annotator-example.png){ align=center width="800" }

    </div>

=== "Mask"

    ```python
    >>> import supervision as sv

    >>> image = ...
    >>> detections = sv.Detections(...)

    >>> mask_annotator = sv.MaskAnnotator()
    >>> annotated_frame = mask_annotator.annotate(
    ...     scene=image.copy(),
    ...     detections=detections
    ... )
    ```

    <div class="result" markdown>

    ![mask-annotator-example](https://media.roboflow.com/supervision-annotator-examples/mask-annotator-example.png){ align=center width="800" }

    </div>

=== "Ellipse"

    ```python
    >>> import supervision as sv

    >>> image = ...
    >>> detections = sv.Detections(...)

    >>> ellipse_annotator = sv.EllipseAnnotator()
    >>> annotated_frame = ellipse_annotator.annotate(
    ...     scene=image.copy(),
    ...     detections=detections
    ... )
    ```

    <div class="result" markdown>

    ![ellipse-annotator-example](https://media.roboflow.com/supervision-annotator-examples/ellipse-annotator-example.png){ align=center width="800" }

    </div>

=== "BoxCorner"

    ```python
    >>> import supervision as sv

    >>> image = ...
    >>> detections = sv.Detections(...)

    >>> corner_annotator = sv.BoxCornerAnnotator()
    >>> annotated_frame = corner_annotator.annotate(
    ...     scene=image.copy(),
    ...     detections=detections
    ... )
    ```

    <div class="result" markdown>

    ![box-corner-annotator-example](https://media.roboflow.com/supervision-annotator-examples/box-corner-annotator-example.png){ align=center width="800" }

    </div>

=== "Circle"

    ```python
    >>> import supervision as sv

    >>> image = ...
    >>> detections = sv.Detections(...)

    >>> circle_annotator = sv.CircleAnnotator()
    >>> annotated_frame = circle_annotator.annotate(
    ...     scene=image.copy(),
    ...     detections=detections
    ... )
    ```

    <div class="result" markdown>

    ![circle-annotator-example](https://media.roboflow.com/supervision-annotator-examples/circle-annotator-example.png){ align=center width="800" }

    </div>

=== "Label"

    ```python
    >>> import supervision as sv

    >>> image = ...
    >>> detections = sv.Detections(...)

    >>> label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
    >>> annotated_frame = label_annotator.annotate(
    ...     scene=image.copy(),
    ...     detections=detections
    ... )
    ```

    <div class="result" markdown>

    ![label-annotator-example](https://media.roboflow.com/supervision-annotator-examples/label-annotator-example-2.png){ align=center width="800" }

    </div>

## BoundingBoxAnnotator

:::supervision.annotators.core.BoundingBoxAnnotator

## MaskAnnotator

:::supervision.annotators.core.MaskAnnotator

## EllipseAnnotator

:::supervision.annotators.core.EllipseAnnotator

## BoxCornerAnnotator

:::supervision.annotators.core.BoxCornerAnnotator

## CircleAnnotator

:::supervision.annotators.core.CircleAnnotator

## LabelAnnotator

:::supervision.annotators.core.LabelAnnotator
