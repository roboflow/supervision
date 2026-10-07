---
comments: true
description: Colour depth and disparity maps over images with sv.DepthAnnotator — Turbo, Viridis, Cividis, Inferno, Magma or grayscale, near objects warm, pixels without depth unpainted.
---

# Depth Annotators

=== "DepthAnnotator"

    ```python
    import supervision as sv

    image = ...
    depth_map = sv.DepthMap(...)

    depth_annotator = sv.DepthAnnotator(opacity=0.65)
    annotated_image = depth_annotator.annotate(
        scene=image.copy(),
        depth_map=depth_map,
    )
    ```

    <div class="result" markdown>

    ![Stereo disparity coloured with sv.DepthAnnotator: a matcher's output over the frame, holes unpainted, and the ground truth alone](https://media.roboflow.com/supervision-annotator-examples/depth-annotator-example.png){ align=center width="800" }

    OpenCV SGBM disparity over the frame (left) and the ground truth (right). Spring dataset by Mehl et al. ([doi:10.18419/darus-3376](https://doi.org/10.18419/darus-3376)) and Spring open movie by Blender Foundation, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); downscaled.

    </div>

<div class="md-typeset">
    <h2><a href="#supervision.depth.annotators.DepthAnnotator">DepthAnnotator</a></h2>
</div>

:::supervision.depth.annotators.DepthAnnotator

<div class="md-typeset">
    <h2><a href="#supervision.depth.colormaps.DepthColormap">DepthColormap</a></h2>
</div>

:::supervision.depth.colormaps.DepthColormap
