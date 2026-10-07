---
comments: true
description: Colour depth, disparity and relative depth maps over images with sv.DepthMap and sv.DepthAnnotator, convert disparity to metres and label objects with their distance.
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
3. [Reading distances and labelling objects](#label-objects-with-their-distance)

## Load a Depth Map

A depth map is a float array with one value per pixel and the kind of value it holds: `"disparity_px"` for a stereo matcher's disparity, `"depth_m"` for metric depth, or `"relative_inverse"` for a monocular model's relative depth.

```python
import numpy as np
import supervision as sv
from PIL import Image

image = sv.pillow_to_cv2(Image.open("<SOURCE_IMAGE_PATH>"))
disparity = np.load("<DISPARITY_NPY_PATH>")  # float32 pixels, left view

depth_map = sv.DepthMap(
    disparity,
    kind="disparity_px",
    camera=sv.DepthCamera(fx_px=1050.0, baseline_m=0.12),
)
```

The camera is optional: with the rig's focal length and baseline, a stereo matcher's disparity gives metric depth too.

Every kind keeps "no depth" explicit: `NaN`, infinities and values at or below 0 (below 0 for relative maps) are pixels the model or matcher could not measure, and `depth_map.valid_mask` marks the rest.

## Colour a Depth Map

```python
depth_annotator = sv.DepthAnnotator(opacity=0.65)
annotated_image = depth_annotator.annotate(image.copy(), depth_map)
```

The defaults:

- `colormap="turbo"` separates the most depth steps. `"viridis"` and `"cividis"` keep their order in grayscale and for colour-blind readers; use them for figures.
- `quantity="disparity"` colours inverse depth, which gives near detail most of the colours. `quantity="depth"` colours metres and needs a metric map or a stereo camera.
- `display_range="auto"` uses the map's 2nd to 98th percentile. A `(low, high)` tuple fixes the range in pixels or metres.
- Pixels without depth stay unpainted, so the image shows through where the model gave up.

To show the depth alone, annotate a blank canvas instead of the image. To paint the pixels without depth in one colour:

```python
height, width = annotated_image.shape[:2]
annotated_image[~depth_map.resize((width, height)).valid_mask] = sv.Color.BLACK.as_bgr()
```

## Label Objects with Their Distance

[measure_detections][supervision.depth.core.DepthMap.measure_detections] stores the median depth inside each mask, or each box without masks, in `detections.data["depth_m"]`, ready for labels drawn with [sv.LabelAnnotator][supervision.annotators.core.LabelAnnotator].

```python
detections = depth_map.measure_detections(detections)

labels = [
    f"{name} {depth:.1f} m"
    for name, depth in zip(detections.data["class_name"], detections.data["depth_m"])
]
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
```

A relative map has no metres, so it fills `detections.data["relative_inverse"]` instead: useful to sort objects from near to far within one image, not to compare images. For one pixel, `depth_map.value_at(x, y)` returns the value or `None` where there is no depth.

## Attribution

The image on this page is frame 96 of sequence 0021 of the Spring dataset by Lukas Mehl, Jenny Schmalfuss, Azin Jahedi, Yaroslava Nalivayko and Andrés Bruhn, "Spring: A High-Resolution High-Detail Dataset and Benchmark for Scene Flow, Optical Flow and Stereo", CVPR 2023, [doi:10.18419/darus-3376](https://doi.org/10.18419/darus-3376), and of the Spring open movie by Blender Foundation, both licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes made: downscaled to 1280x720, disparity quantised to 1/1024 px, a disparity layer computed with OpenCV StereoSGBM, and coloured with `sv.DepthAnnotator`.
