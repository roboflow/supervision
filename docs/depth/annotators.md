---
comments: true
description: Colour depth and disparity maps over images with sv.DepthAnnotator — Turbo, Viridis, Cividis, Inferno, Magma or grayscale, near objects warm, pixels without depth unpainted.
---

# Depth Annotators

=== "Overlay"

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

=== "Colour-blind safe"

    ```python
    import supervision as sv

    image = ...
    depth_map = sv.DepthMap(...)

    depth_annotator = sv.DepthAnnotator(
        colormap="cividis",
        display_range=(2.0, 120.0),
    )
    annotated_image = depth_annotator.annotate(
        scene=image.copy(),
        depth_map=depth_map,
    )
    ```

<div class="result" markdown>

![Stereo disparity coloured with sv.DepthAnnotator: a matcher's output over the frame, holes unpainted, and the ground truth alone](https://media.roboflow.com/supervision-annotator-examples/depth-annotator-example.png){ align=center width="800" }

Left: OpenCV SGBM disparity over the frame at opacity 0.65; the frame shows through where the matcher found no depth. Right: the ground truth alone, same colour range. Frame 96 of Spring sequence 0021 by Mehl et al. (CVPR 2023, [doi:10.18419/darus-3376](https://doi.org/10.18419/darus-3376)) and the Spring open movie by Blender Foundation, both [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); downscaled to 1280x720, disparity quantised to 1/1024 px, SGBM layer added.

</div>

<div class="md-typeset">
    <h2><a href="#supervision.depth.annotators.DepthAnnotator">DepthAnnotator</a></h2>
</div>

:::supervision.depth.annotators.DepthAnnotator

<div class="md-typeset">
    <h2><a href="#supervision.depth.colormaps.DepthColormap">DepthColormap</a></h2>
</div>

:::supervision.depth.colormaps.DepthColormap
