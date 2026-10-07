---
comments: true
description: Load depth maps from Roboflow Inference, Ultralytics, Hugging Face Transformers or stereo datasets, colour them over images and video with sv.DepthAnnotator, convert disparity to metres and label objects with their distance.
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

    depth_map = sv.DepthMap(
        disparity,
        kind="disparity_px",
        camera=sv.DepthCamera(fx_px=1050.0, baseline_m=0.12),
    )
    ```

    The camera is optional: with the rig's focal length and baseline, a stereo matcher's disparity gives metric depth too.

    Dataset files load directly: `sv.DepthMap.from_png16(path, scale=256, kind="disparity_px")` for KITTI and `sv.DepthMap.from_pfm(path)` for Middlebury and SceneFlow.

`NaN`, infinities and values at or below 0 (below 0 for relative maps) are pixels without depth; `depth_map.valid_mask` marks the rest.

## Colour a Depth Map

```python
depth_annotator = sv.DepthAnnotator(opacity=0.65)
annotated_image = depth_annotator.annotate(image.copy(), depth_map)
```

- `colormap="turbo"` separates the most depth steps; `"viridis"` and `"cividis"` keep their order in grayscale and for colour-blind readers.
- `quantity="disparity"` colours inverse depth, which gives near detail most of the colours; `quantity="depth"` colours metres and needs a metric map or a stereo camera.
- `display_range="auto"` uses the map's 2nd to 98th percentile; a `(low, high)` tuple fixes the range in the coloured unit: pixels for disparity, metres for depth, and 1 / metres for a metric map without a camera coloured as disparity (the default). `sv.DepthClipRange` holds one range across a video.

To show the depth alone, annotate a blank canvas instead of the image. To paint the pixels without depth in one colour:

```python
height, width = annotated_image.shape[:2]
annotated_image[~depth_map.resize((width, height)).valid_mask] = sv.Color.BLACK.as_bgr()
```

## Label Objects with Their Distance

[measure_detections][supervision.depth.core.DepthMap.measure_detections] stores the median depth inside each mask, or each box without masks, in `detections.data["depth_m"]`, ready for labels drawn with [sv.LabelAnnotator][supervision.annotators.core.LabelAnnotator].

```python
from inference import get_model

model = get_model(model_id="rfdetr-small")
detections = sv.Detections.from_inference(model.infer(image)[0])
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
import numpy as np
import supervision as sv


def estimate_depth(frame: np.ndarray) -> sv.DepthMap:
    raise NotImplementedError("return your depth model's map for this frame")


source = "<SOURCE_VIDEO_PATH>"
depth_maps = [estimate_depth(frame) for frame in sv.get_video_frames_generator(source)]
clip_range = sv.DepthClipRange.from_depth_maps(depth_maps)

depth_annotator = sv.DepthAnnotator(display_range=clip_range, opacity=0.65)
frames = sv.get_video_frames_generator(source)

with sv.VideoSink("<TARGET_VIDEO_PATH>", sv.VideoInfo.from_video_path(source)) as sink:
    for frame, depth_map in zip(frames, depth_maps):
        sink.write_frame(depth_annotator.annotate(frame, depth_map))
```

`DepthClipRange.from_depth_maps` reads a generator too, so for long clips you can estimate depth twice instead of holding every map. A locked range keeps the colour scale fixed: colours stay put only where the depth values are steady, as in ground truth or calibrated stereo, and a model's own frame-to-frame wobble becomes more visible.

## Attribution

Image: frame 96 of sequence 0021 of the Spring dataset (Mehl et al., CVPR 2023, [doi:10.18419/darus-3376](https://doi.org/10.18419/darus-3376)) and the Spring open movie by Blender Foundation, both [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); downscaled, with an OpenCV StereoSGBM disparity layer coloured by `sv.DepthAnnotator`.
