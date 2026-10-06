---
comments: true
description: Load depth maps from Roboflow Inference, Ultralytics YOLO26 depth, Hugging Face or stereo datasets and colour them over images with sv.DepthAnnotator.
authors:
  - name: Caio Viotti
    role: Roboflow
    github: https://github.com/cfviotti
date_modified: 2026-10-05
---

# Visualize Depth Maps

A depth map holds a distance for every pixel of an image: stereo disparity in pixels, metric depth in metres, or the relative depth a monocular model such as Depth Anything predicts, normalised so larger is nearer. [sv.DepthMap][supervision.depth.core.DepthMap] keeps any of the three with what it measures, and [sv.DepthAnnotator][supervision.depth.annotators.DepthAnnotator] colours it over the image, near objects warm and far ones cool, leaving pixels without depth unpainted.

![Stereo disparity coloured with sv.DepthAnnotator](https://media.roboflow.com/supervision-annotator-examples/depth-annotator-example.png){ align=center width="800" }

This guide covers:

1. [Loading a depth map](#load-a-depth-map)
2. [Colouring it](#colour-a-depth-map)

## Load a Depth Map

=== "Inference"

    Roboflow depth models return a map normalised per image, 1 for the nearest pixel and 0 for the farthest. It loads as `relative_inverse`: good for pictures, not for distances.

    ```python
    import supervision as sv
    from inference import get_model
    from supervision import _cv2 as cv2

    image = cv2.imread("<SOURCE_IMAGE_PATH>")
    model = get_model(model_id="depth-anything-v3/small")

    depth_map = sv.DepthMap.from_inference(model.infer(image)[0])
    ```

=== "Ultralytics"

    YOLO26 depth models predict metric depth in metres.

    ```python
    import supervision as sv
    from ultralytics import YOLO

    model = YOLO("yolo26n-depth.pt")
    result = model("<SOURCE_IMAGE_PATH>")[0]

    depth_map = sv.DepthMap.from_ultralytics(result)
    ```

=== "Transformers"

    The pipeline does not say what its model predicts, so name the kind.

    ```python
    import supervision as sv
    from transformers import pipeline

    estimator = pipeline(
        "depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf"
    )
    result = estimator("<SOURCE_IMAGE_PATH>")

    depth_map = sv.DepthMap.from_transformers(result, kind="relative_inverse")
    ```

=== "Stereo"

    A stereo matcher's disparity, or any other float array, loads with the kind of value it holds.

    ```python
    import numpy as np
    import supervision as sv

    disparity = np.load("<DISPARITY_NPY_PATH>")  # float32 pixels, left view

    depth_map = sv.DepthMap(disparity, kind="disparity_px")
    ```

    Dataset files load directly: `sv.DepthMap.from_png16(path, scale=256, kind="disparity_px")` for KITTI and `sv.DepthMap.from_pfm(path)` for Middlebury and SceneFlow.

Every kind keeps "no depth" explicit: `NaN`, infinities and values at or below 0 (below 0 for relative maps) are pixels the model or matcher could not measure, and `depth_map.valid_mask` marks the rest.

## Colour a Depth Map

```python
depth_annotator = sv.DepthAnnotator(opacity=0.65)
annotated_image = depth_annotator.annotate(image.copy(), depth_map)
```

The defaults:

- `colormap="turbo"` separates the most depth steps. `"viridis"` and `"cividis"` keep their order in grayscale and for colour-blind readers; use them for figures.
- `display_range="auto"` uses the map's 2nd to 98th percentile. A `(low, high)` tuple fixes the range.
- A metric map is coloured as inverse depth, which gives near detail most of the colours.
- Pixels without depth stay unpainted, so the image shows through where the model gave up.

To show the depth alone, annotate a blank canvas instead of the image. To paint the pixels without depth in one colour, for a map the size of the image:

```python
annotated_image[~depth_map.valid_mask] = sv.Color.BLACK.as_bgr()
```

## Attribution

The image on this page is frame 96 of sequence 0021 of the Spring dataset by Lukas Mehl, Jenny Schmalfuss, Azin Jahedi, Yaroslava Nalivayko and Andrés Bruhn, "Spring: A High-Resolution High-Detail Dataset and Benchmark for Scene Flow, Optical Flow and Stereo", CVPR 2023, [doi:10.18419/darus-3376](https://doi.org/10.18419/darus-3376), and of the Spring open movie by Blender Foundation, both licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes made: downscaled to 1280x720, disparity quantised to 1/1024 px, a disparity layer computed with OpenCV StereoSGBM, and coloured with `sv.DepthAnnotator`.
