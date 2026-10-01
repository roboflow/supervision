---
comments: true
description: Load depth maps from Roboflow Inference, Ultralytics YOLO26 depth, Hugging Face or a stereo matcher, colour them over images and video, label objects with their distance, and write clips supervision-js plays in the browser.
authors:
  - name: Caio Viotti
    role: Roboflow
    github: https://github.com/cfviotti
date_modified: 2026-10-01
---

# Visualize Depth Maps

A depth map holds a distance for every pixel of an image: stereo disparity in pixels, metric depth in metres, or the relative inverse depth a monocular model such as Depth Anything predicts. [sv.DepthMap][supervision.depth.core.DepthMap] keeps any of the three with what it measures, and [sv.DepthAnnotator][supervision.depth.annotators.DepthAnnotator] colours it over the image, near objects warm and far ones cool, leaving pixels without depth unpainted.

![Stereo disparity coloured with sv.DepthAnnotator](https://media.roboflow.com/supervision-annotator-examples/depth-annotator-example.png){ align=center width="800" }

This guide covers:

1. [Loading a depth map](#load-a-depth-map)
2. [Colouring it](#colour-a-depth-map)
3. [Reading distances and labelling objects](#label-objects-with-their-distance)
4. [Colouring a video with one range](#colour-a-video-with-one-range)
5. [Saving depth for the browser](#save-depth-for-supervision-js)

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

    A stereo matcher's disparity, with the rig's focal length and baseline, gives metric depth too.

    ```python
    import numpy as np
    import supervision as sv

    disparity = np.load("<DISPARITY_NPY_PATH>")  # float32 pixels, left view

    depth_map = sv.DepthMap(
        disparity,
        kind="disparity_px",
        camera=sv.DepthCamera(fx_px=1050.0, baseline_m=0.12),
    )
    ```

    Dataset files load directly: `sv.DepthMap.from_png16(path, scale=256, kind="disparity_px")` for KITTI, `sv.DepthMap.from_pfm(path)` for Middlebury and SceneFlow, `sv.DepthMap.from_npy(path, kind=...)` for arrays.

Every kind keeps "no depth" explicit: `NaN`, infinities and values at or below 0 (below 0 for relative maps) are pixels the model or matcher could not measure, and `depth_map.valid_mask` marks the rest.

## Colour a Depth Map

```python
depth_annotator = sv.DepthAnnotator(opacity=0.65)
annotated_image = depth_annotator.annotate(image.copy(), depth_map)
```

The defaults are the ones supervision-js uses, so a map looks the same in a notebook and in the browser:

- `colormap="turbo"` separates the most depth steps. `"viridis"` and `"cividis"` keep their order in grayscale and for colour-blind readers; use them for figures.
- `quantity="disparity"` colours inverse depth, which gives near detail most of the colours. `quantity="depth"` colours metres and needs a metric map or a stereo camera.
- `display_range="clip"` uses the map's own range when it has one and its 2nd to 98th percentile otherwise. A `(low, high)` tuple fixes the range in pixels or metres.
- Pixels without depth stay unpainted, so the image shows through where the model gave up. Pass `no_depth_color=sv.Color.BLACK` to paint them.

To draw a colour bar that matches, build a matplotlib colormap from the same table:

```python
from matplotlib.colors import ListedColormap

turbo = ListedColormap(sv.DepthColormap.TURBO.rgb_lut() / 255)
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

## Colour a Video with One Range

Colouring each frame with its own range makes a still wall change colour whenever something enters the frame. Compute one range for the whole clip in a first pass, then colour every frame with it:

```python
import supervision as sv

source = "<SOURCE_VIDEO_PATH>"
depth_maps = [estimate_depth(frame) for frame in sv.get_video_frames_generator(source)]
clip_range = sv.DepthClipRange.from_depth_maps(depth_maps)

depth_annotator = sv.DepthAnnotator(display_range=clip_range, opacity=0.65)
frames = sv.get_video_frames_generator(source)

with sv.VideoSink("<TARGET_VIDEO_PATH>", sv.VideoInfo.from_video_path(source)) as sink:
    for frame, depth_map in zip(frames, depth_maps):
        sink.write_frame(depth_annotator.annotate(frame, depth_map))
```

`DepthClipRange.from_depth_maps` reads a generator too, so for long clips you can estimate depth twice instead of holding every map. Relative maps from monocular models change scale from frame to frame by design, so a locked range still flickers with them; metric and stereo maps do not.

## Save Depth for supervision-js

[supervision-js](https://github.com/roboflow/supervision-js) draws depth in the browser and reads exact values under the pointer. `save` writes the format it loads, a `depth.json` manifest and a 16-bit PNG:

```python
depth_map.save("depth/depth.json")  # also writes depth/depth.png
depth_map = sv.DepthMap.load("depth/depth.json")
```

For video, [sv.DepthSink][supervision.depth.sink.DepthSink] writes one exact PNG per video frame, an 8-bit preview video the browser plays, and the clip manifest:

```python
video_info = sv.VideoInfo.from_video_path(source)
clip_range = sv.DepthClipRange.from_depth_maps(depth_maps)

with sv.DepthSink("left-depth", video_info, clip_range, preview=False) as sink:
    for depth_map in depth_maps:
        sink.write_depth_map(depth_map)
```

The preview video is defined for disparity only, so this example passes `preview=False`, which metric and relative maps need; supervision-js then draws exact depth while playback rests. For stereo disparity maps, drop `preview=False` to also write the preview.

## Attribution

The image on this page is frame 96 of sequence 0021 of the Spring dataset by Lukas Mehl, Jenny Schmalfuss, Azin Jahedi, Yaroslava Nalivayko and Andrés Bruhn, "Spring: A High-Resolution High-Detail Dataset and Benchmark for Scene Flow, Optical Flow and Stereo", CVPR 2023, [doi:10.18419/darus-3376](https://doi.org/10.18419/darus-3376), and of the Spring open movie by Blender Foundation, both licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes made: downscaled to 1280x720, disparity quantised to 1/1024 px, a disparity layer computed with OpenCV StereoSGBM, and coloured with `sv.DepthAnnotator`.
