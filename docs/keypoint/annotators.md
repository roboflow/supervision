---
comments: true
---

# Annotators

=== "VertexAnnotator"

    ```python
    import supervision as sv

    image = ...
    key_points = sv.KeyPoints(...)

    vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.GREEN,
        radius=10,
    )
    annotated_frame = vertex_annotator.annotate(
        scene=image.copy(),
        key_points=key_points,
    )
    ```

    <div class="result" markdown>

    ![vertex-annotator-example](https://media.roboflow.com/supervision-annotator-examples/vertex-annotator-example.png){ align=center width="800" }

    </div>

=== "EdgeAnnotator"

    ```python
    import supervision as sv

    image = ...
    key_points = sv.KeyPoints(...)

    edge_annotator = sv.EdgeAnnotator(
        color=sv.Color.GREEN,
        thickness=5,
    )
    annotated_frame = edge_annotator.annotate(
        scene=image.copy(),
        key_points=key_points,
    )
    ```

    <div class="result" markdown>

    ![edge-annotator-example](https://media.roboflow.com/supervision-annotator-examples/edge-annotator-example.png){ align=center width="800" }

    </div>

=== "VertexLabelAnnotator"

    ```python
    import supervision as sv

    image = ...
    key_points = sv.KeyPoints(...)

    vertex_label_annotator = sv.VertexLabelAnnotator(
        color=sv.Color.GREEN,
        text_color=sv.Color.BLACK,
        border_radius=5,
    )
    annotated_frame = vertex_label_annotator.annotate(
        scene=image.copy(),
        key_points=key_points,
    )
    ```

    <div class="result" markdown>

    ![vertex-label-annotator-example](https://media.roboflow.com/supervision-annotator-examples/vertex-label-annotator-example.png){ align=center width="800" }

    </div>

=== "VertexEllipseAnnotator"

    ```python
    import numpy as np
    import supervision as sv

    image = ...
    key_points = sv.KeyPoints(...)

    # covariance shape: (N, K, 2, 2) — pixel-space covariance per keypoint
    covariance = np.zeros((len(key_points), key_points.xy.shape[1], 2, 2), dtype=np.float32)
    key_points.data["covariance"] = covariance

    ellipse_annotator = sv.VertexEllipseAnnotator(
        color=sv.Color.GREEN,
        thickness=2,
        sigma=2.0,
    )
    annotated_frame = ellipse_annotator.annotate(
        scene=image.copy(),
        key_points=key_points,
    )
    ```

<div class="md-typeset">
  <h2><a href="#supervision.key_points.annotators.VertexAnnotator">VertexAnnotator</a></h2>
</div>

:::supervision.key_points.annotators.VertexAnnotator

<div class="md-typeset">
  <h2><a href="#supervision.key_points.annotators.EdgeAnnotator">EdgeAnnotator</a></h2>
</div>

:::supervision.key_points.annotators.EdgeAnnotator

<div class="md-typeset">
  <h2><a href="#supervision.key_points.annotators.VertexLabelAnnotator">VertexLabelAnnotator</a></h2>
</div>

:::supervision.key_points.annotators.VertexLabelAnnotator

<div class="md-typeset">
  <h2><a href="#supervision.key_points.annotators.VertexEllipseAnnotator">VertexEllipseAnnotator</a></h2>
</div>

:::supervision.key_points.annotators.VertexEllipseAnnotator
