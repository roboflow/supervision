---
comments: true
description: Colour depth, disparity and relative depth maps over images with sv.DepthMap and sv.DepthAnnotator.
authors:
  - name: Caio Viotti
    role: Roboflow
    github: https://github.com/cfviotti
date_modified: 2026-10-08
---

# Visualize Depth Maps

A depth map holds a distance for every pixel of an image: stereo disparity in pixels, metric depth in metres, or a monocular model's relative depth, larger meaning nearer. [sv.DepthMap][supervision.depth.core.DepthMap] holds it with what it measures, and [sv.DepthAnnotator][supervision.depth.annotators.DepthAnnotator] colours it over the image, near objects warm, leaving pixels without depth unpainted.

![Stereo disparity coloured with sv.DepthAnnotator](https://media.roboflow.com/supervision-annotator-examples/depth-annotator-example.png){ align=center width="800" }

## Load a Depth Map

Pass a float array with one value per pixel and the kind of value it holds: `"disparity_px"`, `"depth_m"` or `"relative_inverse"`.

```python
import numpy as np
import supervision as sv
from PIL import Image

image = sv.pillow_to_cv2(Image.open("<SOURCE_IMAGE_PATH>"))
disparity = np.load("<DISPARITY_NPY_PATH>")  # float32 pixels, left view

depth_map = sv.DepthMap(disparity, kind="disparity_px")
```

`NaN`, infinities and values at or below 0 (below 0 for relative maps) are pixels without depth; `depth_map.valid_mask` marks the rest.

For `"relative_inverse"`, larger values are nearer and 0 is the farthest valid value, so set pixels without depth to `NaN`. Invert a relative map that grows with distance, such as Depth Anything V3's, before wrapping it; negating it instead would leave every pixel negative, and so without depth:

```python
relative_depth = np.load("<RELATIVE_DEPTH_NPY_PATH>")  # float32, grows with distance
relative_depth[relative_depth <= 0] = np.nan  # pixels without depth
depth_map = sv.DepthMap(1 / relative_depth, kind="relative_inverse")
```

## Colour a Depth Map

```python
depth_annotator = sv.DepthAnnotator(opacity=0.65)
annotated_image = depth_annotator.annotate(image.copy(), depth_map)
```

- `colormap="turbo"` separates the most depth steps; `"viridis"` and `"cividis"` keep their order in grayscale and for colour-blind readers.
- `display_range="auto"` uses the map's 2nd to 98th percentile; a `(low, high)` tuple fixes the range in the map's own unit, such as `(1.0, 10.0)` metres for a metric map.
- A metric map is coloured as inverse depth, which gives near detail most of the colours.

To show the depth alone, annotate a blank canvas instead of the image. To paint the pixels without depth in one colour, for a map the size of the image:

```python
annotated_image[~depth_map.valid_mask] = sv.Color.BLACK.as_bgr()
```

## Attribution

Image: frame 96 of sequence 0021 of the Spring dataset (Mehl et al., CVPR 2023, [doi:10.18419/darus-3376](https://doi.org/10.18419/darus-3376)) and the Spring open movie by Blender Foundation, both [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); downscaled, with an OpenCV StereoSGBM disparity layer coloured by `sv.DepthAnnotator`.
