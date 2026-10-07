---
comments: true
description: Load depth maps from Roboflow Inference, Ultralytics, Hugging Face Transformers or stereo datasets and colour them over images with sv.DepthAnnotator.
authors:
  - name: Caio Viotti
    role: Roboflow
    github: https://github.com/cfviotti
date_modified: 2026-10-05
---

# Visualize Depth Maps

A depth map holds a distance for every pixel of an image: stereo disparity in pixels, metric depth in metres, or a monocular model's relative depth, larger meaning nearer. [sv.DepthMap][supervision.depth.core.DepthMap] holds it with what it measures, and [sv.DepthAnnotator][supervision.depth.annotators.DepthAnnotator] colours it over the image, near objects warm, leaving pixels without depth unpainted.

![Stereo disparity coloured with sv.DepthAnnotator](https://media.roboflow.com/supervision-annotator-examples/depth-annotator-example.png){ align=center width="800" }

## Load a Depth Map

Open the image, then load its depth map from a model or a file.

```python
import supervision as sv
from PIL import Image

image = sv.pillow_to_cv2(Image.open("<SOURCE_IMAGE_PATH>"))
```

=== "Inference"

    Depth models served by Roboflow Inference (Depth Anything, YOLO26 depth) return a map normalised per image, 1 for the nearest pixel and 0 for the farthest. It loads as `relative_inverse`: good for pictures, not for distances.

    ```python
    from inference import get_model

    model = get_model(model_id="depth-anything-v3/small")

    depth_map = sv.DepthMap.from_inference(model.infer(image)[0])
    ```

=== "Ultralytics"

    YOLO26 depth models predict metric depth in metres.

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26n-depth.pt")

    depth_map = sv.DepthMap.from_ultralytics(model(image)[0])
    ```

=== "Transformers"

    The pipeline does not say what its model predicts, so name the kind.

    ```python
    from transformers import pipeline

    estimator = pipeline(
        "depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf"
    )
    result = estimator(sv.cv2_to_pillow(image))

    depth_map = sv.DepthMap.from_transformers(result, kind="relative_inverse")
    ```

=== "Stereo"

    A stereo matcher's disparity, or any other float array, loads with the kind of value it holds.

    ```python
    import numpy as np

    disparity = np.load("<DISPARITY_NPY_PATH>")  # float32 pixels, left view

    depth_map = sv.DepthMap(disparity, kind="disparity_px")
    ```

    Dataset files load directly: `sv.DepthMap.from_png16(path, scale=256, kind="disparity_px")` for KITTI and `sv.DepthMap.from_pfm(path)` for Middlebury and SceneFlow.

`NaN`, infinities and values at or below 0 (below 0 for relative maps) are pixels without depth; `depth_map.valid_mask` marks the rest.

## Colour a Depth Map

```python
depth_annotator = sv.DepthAnnotator(opacity=0.65)
annotated_image = depth_annotator.annotate(image.copy(), depth_map)
```

- `colormap="turbo"` separates the most depth steps; `"viridis"` and `"cividis"` keep their order in grayscale and for colour-blind readers.
- `display_range="auto"` uses the map's 2nd to 98th percentile; a `(low, high)` tuple fixes the range.
- A metric map is coloured as inverse depth, which gives near detail most of the colours.

To show the depth alone, annotate a blank canvas instead of the image. To paint the pixels without depth in one colour, for a map the size of the image:

```python
annotated_image[~depth_map.valid_mask] = sv.Color.BLACK.as_bgr()
```

## Attribution

Image: frame 96 of sequence 0021 of the Spring dataset (Mehl et al., CVPR 2023, [doi:10.18419/darus-3376](https://doi.org/10.18419/darus-3376)) and the Spring open movie by Blender Foundation, both [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); downscaled, with an OpenCV StereoSGBM disparity layer coloured by `sv.DepthAnnotator`.
