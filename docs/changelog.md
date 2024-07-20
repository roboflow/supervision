### 0.22.0 <small>Jul 12, 2024</small>

- Added [#1326](https://github.com/roboflow/supervision/pull/1326): [`sv.DetectionsDataset`](https://supervision.roboflow.com/latest/datasets/core/#supervision.dataset.core.DetectionDataset) and [`sv.ClassificationDataset`](https://supervision.roboflow.com/latest/datasets/core/#supervision.dataset.core.ClassificationDataset) allowing to load the images into memory only when necessary (lazy loading).

!!! failure "Deprecated"

    Constructing `DetectionDataset` with parameter `images` as `Dict[str, np.ndarray]` is deprecated and will be removed in `supervision-0.26.0`. Please pass a list of paths `List[str]` instead.

!!! failure "Deprecated"

    The `DetectionDataset.images` property is deprecated and will be removed in `supervision-0.26.0`. Please loop over images with `for path, image, annotation in dataset:`, as that does not require loading all images into memory.

```python
import roboflow
from roboflow import Roboflow
import supervision as sv

roboflow.login()
rf = Roboflow()

project = rf.workspace(<WORKSPACE_ID>).project(<PROJECT_ID>)
dataset = project.version(<PROJECT_VERSION>).download("coco")

ds_train = sv.DetectionDataset.from_coco(
    images_directory_path=f"{dataset.location}/train",
    annotations_path=f"{dataset.location}/train/_annotations.coco.json",
)

path, image, annotation = ds_train[0]
    # loads image on demand

for path, image, annotation in ds_train:
    # loads image on demand
```

- Added [#1296](https://github.com/roboflow/supervision/pull/1296): [`sv.Detections.from_lmm`](https://supervision.roboflow.com/latest/detection/core/#supervision.detection.core.Detections.from_lmm) now supports parsing results from the [Florence 2](https://huggingface.co/microsoft/Florence-2-large) model, extending the capability to handle outputs from this Large Multimodal Model (LMM). This includes detailed object detection, OCR with region proposals, segmentation, and more. Find out more in our [Colab notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-florence-2-on-detection-dataset.ipynb).

- Added [#1232](https://github.com/roboflow/supervision/pull/1232) to support keypoint detection with Mediapipe. Both [legacy](https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb) and [modern](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python) pipelines are supported. See [`sv.KeyPoints.from_mediapipe`](https://supervision.roboflow.com/latest/keypoint/core/#supervision.keypoint.core.KeyPoints.from_mediapipe) for more.

- Added [#1316](https://github.com/roboflow/supervision/pull/1316): [`sv.KeyPoints.from_mediapipe`](https://supervision.roboflow.com/latest/keypoint/core/#supervision.keypoint.core.KeyPoints.from_mediapipe) extended to support FaceMesh from Mediapipe. This enhancement allows for processing both face landmarks from `FaceLandmarker`, and legacy results from `FaceMesh`.

- Added [#1310](https://github.com/roboflow/supervision/pull/1310): [`sv.KeyPoints.from_detectron2`](https://supervision.roboflow.com/latest/keypoint/core/#supervision.keypoint.core.KeyPoints.from_detectron2) is a new `KeyPoints` method, adding support for extracting keypoints from the popular [Detectron 2](https://github.com/facebookresearch/detectron2) platform.

- Added [#1300](https://github.com/roboflow/supervision/pull/1300): [`sv.Detections.from_detectron2`](https://supervision.roboflow.com/latest/detection/core/#supervision.detection.core.Detections.from_detectron2) now supports segmentation models detectron2. The resulting masks can be used with [`sv.MaskAnnotator`](https://supervision.roboflow.com/latest/annotators/#supervision.annotators.core.MaskAnnotator) for displaying annotations.

```python
import supervision as sv
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import cv2

image = cv2.imread(<SOURCE_IMAGE_PATH>)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

result = predictor(image)
detections = sv.Detections.from_detectron2(result)

mask_annotator = sv.MaskAnnotator()
annotated_frame = mask_annotator.annotate(scene=image.copy(), detections=detections)
```

- Added [#1277](https://github.com/roboflow/supervision/pull/1277): if you provide a font that supports symbols of a language, [`sv.RichLabelAnnotator`](https://supervision.roboflow.com/latest/detection/annotators/#supervision.annotators.core.LabelAnnotator.annotate) will draw them on your images.
  - Various other annotators have been revised to ensure proper in-place functionality when used with `numpy` arrays. Additionally, we fixed a bug where `sv.ColorAnnotator` was filling boxes with solid color when used in-place.

```python
import cv2
import supervision as sv
import

image = cv2.imread(<SOURCE_IMAGE_PATH>)

model = get_model(model_id="yolov8n-640")
results = model.infer(image)[0]
detections = sv.Detections.from_inference(results)

rich_label_annotator = sv.RichLabelAnnotator(font_path=<TTF_FONT_PATH>)
annotated_image = rich_label_annotator.annotate(scene=image.copy(), detections=detections)
```

- Added [#1227](https://github.com/roboflow/supervision/pull/1227): Added support for loading Oriented Bounding Boxes dataset in YOLO format.

```python
import supervision as sv

train_ds = sv.DetectionDataset.from_yolo(
    images_directory_path="/content/dataset/train/images",
    annotations_directory_path="/content/dataset/train/labels",
    data_yaml_path="/content/dataset/data.yaml",
    is_obb=True
)

_, image, detections in train_ds[0]

obb_annotator = OrientedBoxAnnotator()
annotated_image = obb_annotator.annotate(scene=image.copy(), detections=detections)
```

- Fixed [#1312](https://github.com/roboflow/supervision/pull/1312): Fixed [`CropAnnotator`](https://supervision.roboflow.com/latest/detection/annotators/#supervision.annotators.core.TraceAnnotator.annotate).

!!! failure "Removed"

    `BoxAnnotator` was removed, however `BoundingBoxAnnotator` has been renamed to `BoxAnnotator`. Use a combination of [`BoxAnnotator`](https://supervision.roboflow.com/latest/detection/annotators/#supervision.annotators.core.BoxAnnotator) and [`LabelAnnotator`](https://supervision.roboflow.com/latest/detection/annotators/#supervision.annotators.core.LabelAnnotator) to simulate old `BoundingBox` behavior.

!!! failure "Deprecated"

    The name `BoundingBoxAnnotator` has been deprecated and will be removed in `supervision-0.26.0`. It has been renamed to [`BoxAnnotator`](https://supervision.roboflow.com/latest/detection/annotators/#supervision.annotators.core.BoxAnnotator).

- Added [#975](https://github.com/roboflow/supervision/pull/975) 📝 New Cookbooks: serialize detections into [json](https://github.com/roboflow/supervision/blob/de896189b83a1f9434c0a37dd9192ee00d2a1283/docs/notebooks/serialise-detections-to-json.ipynb) and [csv](https://github.com/roboflow/supervision/blob/de896189b83a1f9434c0a37dd9192ee00d2a1283/docs/notebooks/serialise-detections-to-csv.ipynb).

- Added [#1290](https://github.com/roboflow/supervision/pull/1290): Mostly an internal change, our file utility function now support both `str` and `pathlib` paths.

- Added [#1340](https://github.com/roboflow/supervision/pull/1340): Two new methods for converting between bounding box formats - [`xywh_to_xyxy`](https://supervision.roboflow.com/latest/detection/utils/#supervision.detection.utils.xywh_to_xyxy) and [`xcycwh_to_xyxy`](https://supervision.roboflow.com/latest/detection/utils/#supervision.detection.utils.xcycwh_to_xyxy)

!!! failure "Removed"

    `from_roboflow` method has been removed due to deprecation. Use [from_inference](https://supervision.roboflow.com/latest/detection/core/#supervision.detection.core.Detections.from_inference) instead.

!!! failure "Removed"

    `Color.white()` has been removed due to deprecation. Use `color.WHITE` instead.

!!! failure "Removed"

    `Color.black()` has been removed due to deprecation. Use `color.BLACK` instead.

!!! failure "Removed"

    `Color.red()` has been removed due to deprecation. Use `color.RED` instead.

!!! failure "Removed"

    `Color.green()` has been removed due to deprecation. Use `color.GREEN` instead.

!!! failure "Removed"

    `Color.blue()` has been removed due to deprecation. Use `color.BLUE` instead.

!!! failure "Removed"

    `ColorPalette.default()` has been removed due to deprecation. Use [ColorPalette.DEFAULT](https://supervision.roboflow.com/latest/utils/draw/#supervision.draw.color.ColorPalette.DEFAULT) instead.

!!! failure "Removed"

    `FPSMonitor.__call__` has been removed due to deprecation. Use the attribute [FPSMonitor.fps](https://supervision.roboflow.com/latest/utils/video/#supervision.utils.video.FPSMonitor.fps) instead.

### 0.21.0 <small>Jun 5, 2024</small>

- Added [#500](https://github.com/roboflow/supervision/pull/500): [`sv.Detections.with_nmm`](https://supervision.roboflow.com/latest/detection/core/#supervision.detection.core.Detections.with_nmm) to perform non-maximum merging on the current set of object detections.

- Added [#1221](https://github.com/roboflow/supervision/pull/1221): [`sv.Detections.from_lmm`](https://supervision.roboflow.com/latest/detection/core/#supervision.detection.core.Detections.from_lmm) allowing to parse Large Multimodal Model (LMM) text result into [`sv.Detections`](https://supervision.roboflow.com/latest/detection/core/) object. For now `from_lmm` supports only [PaliGemma](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-paligemma-on-detection-dataset.ipynb) result parsing.

```python
import supervision as sv

paligemma_result = "<loc0256><loc0256><loc0768><loc0768> cat"
detections = sv.Detections.from_lmm(
    sv.LMM.PALIGEMMA,
    paligemma_result,
    resolution_wh=(1000, 1000),
    classes=['cat', 'dog']
)
detections.xyxy
# array([[250., 250., 750., 750.]])

detections.class_id
# array([0])
```

- Added [#1236](https://github.com/roboflow/supervision/pull/1236): [`sv.VertexLabelAnnotator`](https://supervision.roboflow.com/latest/keypoint/annotators/#supervision.keypoint.annotators.EdgeAnnotator.annotate) allowing to annotate every vertex of a keypoint skeleton with custom text and color.

```python
import supervision as sv

image = ...
key_points = sv.KeyPoints(...)

edge_annotator = sv.EdgeAnnotator(
    color=sv.Color.GREEN,
    thickness=5
)
annotated_frame = edge_annotator.annotate(
    scene=image.copy(),
    key_points=key_points
)
```

- Added [#1147](https://github.com/roboflow/supervision/pull/1147): [`sv.KeyPoints.from_inference`](https://supervision.roboflow.com/develop/keypoint/core/#supervision.keypoint.core.KeyPoints.from_inference) allowing to create [`sv.KeyPoints`](https://supervision.roboflow.com/develop/keypoint/core/#supervision.keypoint.core.KeyPoints) from [Inference](https://github.com/roboflow/inference) result.

- Added [#1138](https://github.com/roboflow/supervision/pull/1138): [`sv.KeyPoints.from_yolo_nas`](https://supervision.roboflow.com/develop/keypoint/core/#supervision.keypoint.core.KeyPoints.from_yolo_nas) allowing to create [`sv.KeyPoints`](https://supervision.roboflow.com/develop/keypoint/core/#supervision.keypoint.core.KeyPoints) from [YOLO-NAS](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md) result.

- Added [#1163](https://github.com/roboflow/supervision/pull/1163): [`sv.mask_to_rle`](https://supervision.roboflow.com/develop/datasets/utils/#supervision.dataset.utils.rle_to_mask) and [`sv.rle_to_mask`](https://supervision.roboflow.com/develop/datasets/utils/#supervision.dataset.utils.rle_to_mask) allowing for easy conversion between mask and rle formats.

- Changed [#1236](https://github.com/roboflow/supervision/pull/1236): [`sv.InferenceSlicer`](https://supervision.roboflow.com/develop/detection/tools/inference_slicer/) allowing to select overlap filtering strategy (`NONE`, `NON_MAX_SUPPRESSION` and `NON_MAX_MERGE`).

- Changed [#1178](https://github.com/roboflow/supervision/pull/1178): [`sv.InferenceSlicer`](https://supervision.roboflow.com/develop/detection/tools/inference_slicer/) adding instance segmentation model support.

```python
import cv2
import numpy as np
import supervision as sv
from inference import get_model

model = get_model(model_id="yolov8x-seg-640")
image = cv2.imread(<SOURCE_IMAGE_PATH>)

def callback(image_slice: np.ndarray) -> sv.Detections:
    results = model.infer(image_slice)[0]
    return sv.Detections.from_inference(results)

slicer = sv.InferenceSlicer(callback = callback)
detections = slicer(image)

mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = mask_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)
```

- Changed [#1228](https://github.com/roboflow/supervision/pull/1228): [`sv.LineZone`](https://supervision.roboflow.com/develop/detection/tools/line_zone/) making it 10-20 times faster, depending on the use case.

- Changed [#1163](https://github.com/roboflow/supervision/pull/1163): [`sv.DetectionDataset.from_coco`](https://supervision.roboflow.com/develop/datasets/core/#supervision.dataset.core.DetectionDataset.from_coco) and [`sv.DetectionDataset.as_coco`](https://supervision.roboflow.com/develop/datasets/core/#supervision.dataset.core.DetectionDataset.as_coco) adding support for run-length encoding (RLE) mask format.

### 0.20.0 <small>April 24, 2024</small>

- Added [#1128](https://github.com/roboflow/supervision/pull/1128): [`sv.KeyPoints`](/0.20.0/keypoint/core/#supervision.keypoint.core.KeyPoints) to provide initial support for pose estimation and broader keypoint detection models.

- Added [#1128](https://github.com/roboflow/supervision/pull/1128): [`sv.EdgeAnnotator`](/0.20.0/keypoint/annotators/#supervision.keypoint.annotators.EdgeAnnotator) and [`sv.VertexAnnotator`](/0.20.0/keypoint/annotators/#supervision.keypoint.annotators.VertexAnnotator) to enable rendering of results from keypoint detection models.

```python
import cv2
import supervision as sv
from ultralytics import YOLO

image = cv2.imread(<SOURCE_IMAGE_PATH>)
model = YOLO('yolov8l-pose')

result = model(image, verbose=False)[0]
keypoints = sv.KeyPoints.from_ultralytics(result)

edge_annotators = sv.EdgeAnnotator(color=sv.Color.GREEN, thickness=5)
annotated_image = edge_annotators.annotate(image.copy(), keypoints)
```

- Changed [#1037](https://github.com/roboflow/supervision/pull/1037): [`sv.LabelAnnotator`](/0.20.0/annotators/#supervision.annotators.core.LabelAnnotator) by adding an additional `corner_radius` argument that allows for rounding the corners of the bounding box.

- Changed [#1109](https://github.com/roboflow/supervision/pull/1109): [`sv.PolygonZone`](/0.20.0/detection/tools/polygon_zone/#supervision.detection.tools.polygon_zone.PolygonZone) such that the `frame_resolution_wh` argument is no longer required to initialize `sv.PolygonZone`.

!!! failure "Deprecated"

    The `frame_resolution_wh` parameter in `sv.PolygonZone` is deprecated and will be removed in `supervision-0.24.0`.

- Changed [#1084](https://github.com/roboflow/supervision/pull/1084): [`sv.get_polygon_center`](/0.20.0/utils/geometry/#supervision.geometry.core.utils.get_polygon_center) to calculate a more accurate polygon centroid.

- Changed [#1069](https://github.com/roboflow/supervision/pull/1069): [`sv.Detections.from_transformers`](/0.20.0/detection/core/#supervision.detection.core.Detections.from_transformers) by adding support for Transformers segmentation models and extract class names values.

```python
import torch
import supervision as sv
from PIL import Image
from transformers import DetrImageProcessor, DetrForSegmentation

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")
model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")

image = Image.open(<SOURCE_IMAGE_PATH>)
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

width, height = image.size
target_size = torch.tensor([[height, width]])
results = processor.post_process_segmentation(
    outputs=outputs, target_sizes=target_size)[0]
detections = sv.Detections.from_transformers(results, id2label=model.config.id2label)

mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

annotated_image = mask_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)
```

- Fixed [#787](https://github.com/roboflow/supervision/pull/787): [`sv.ByteTrack.update_with_detections`](/0.20.0/trackers/#supervision.tracker.byte_tracker.core.ByteTrack.update_with_detections) which was removing segmentation masks while tracking. Now, `ByteTrack` can be used alongside segmentation models.

### 0.19.0 <small>March 15, 2024</small>

- Added [#818](https://github.com/roboflow/supervision/pull/818): [`sv.CSVSink`](/0.19.0/detection/tools/save_detections/#supervision.detection.tools.csv_sink.CSVSink) allowing for the straightforward saving of image, video, or stream inference results in a `.csv` file.

```python
import supervision as sv
from ultralytics import YOLO

model = YOLO(<SOURCE_MODEL_PATH>)
csv_sink = sv.CSVSink(<RESULT_CSV_FILE_PATH>)
frames_generator = sv.get_video_frames_generator(<SOURCE_VIDEO_PATH>)

with csv_sink:
    for frame in frames_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        csv_sink.append(detections, custom_data={<CUSTOM_LABEL>:<CUSTOM_DATA>})
```

- Added [#819](https://github.com/roboflow/supervision/pull/819): [`sv.JSONSink`](/0.19.0/detection/tools/save_detections/#supervision.detection.tools.csv_sink.JSONSink) allowing for the straightforward saving of image, video, or stream inference results in a `.json` file.

```python
import supervision as sv
from ultralytics import YOLO

model = YOLO(<SOURCE_MODEL_PATH>)
json_sink = sv.JSONSink(<RESULT_JSON_FILE_PATH>)
frames_generator = sv.get_video_frames_generator(<SOURCE_VIDEO_PATH>)

with json_sink:
    for frame in frames_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        json_sink.append(detections, custom_data={<CUSTOM_LABEL>:<CUSTOM_DATA>})
```

- Added [#847](https://github.com/roboflow/supervision/pull/847): [`sv.mask_iou_batch`](/0.19.0/detection/utils/#supervision.detection.utils.mask_iou_batch) allowing to compute Intersection over Union (IoU) of two sets of masks.

- Added [#847](https://github.com/roboflow/supervision/pull/847): [`sv.mask_non_max_suppression`](/0.19.0/detection/utils/#supervision.detection.utils.mask_non_max_suppression) allowing to perform Non-Maximum Suppression (NMS) on segmentation predictions.

- Added [#888](https://github.com/roboflow/supervision/pull/888): [`sv.CropAnnotator`](/0.19.0/annotators/#supervision.annotators.core.CropAnnotator) allowing users to annotate the scene with scaled-up crops of detections.

```python
import cv2
import supervision as sv
from inference import get_model

image = cv2.imread(<SOURCE_IMAGE_PATH>)
model = get_model(model_id="yolov8n-640")

result = model.infer(image)[0]
detections = sv.Detections.from_inference(result)

crop_annotator = sv.CropAnnotator()
annotated_frame = crop_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

- Changed [#827](https://github.com/roboflow/supervision/pull/827): [`sv.ByteTrack.reset`](/0.19.0/tracking/#supervision.tracking.ByteTrack.reset) allowing users to clear trackers state, enabling the processing of multiple video files in sequence.

- Changed [#802](https://github.com/roboflow/supervision/pull/802): [`sv.LineZoneAnnotator`](/0.19.0/detection/tools/line_zone/#supervision.detection.line_zone.LineZone) allowing to hide in/out count using `display_in_count` and `display_out_count` properties.

- Changed [#787](https://github.com/roboflow/supervision/pull/787): [`sv.ByteTrack`](/0.19.0/tracking/#supervision.tracking.ByteTrack) input arguments and docstrings updated to improve readability and ease of use.

!!! failure "Deprecated"

    The `track_buffer`, `track_thresh`, and `match_thresh` parameters in `sv.ByterTrack` are deprecated and will be removed in `supervision-0.23.0`. Use `lost_track_buffer,` `track_activation_threshold`, and `minimum_matching_threshold` instead.

- Changed [#910](https://github.com/roboflow/supervision/pull/910): [`sv.PolygonZone`](/0.19.0/detection/tools/polygon_zone/#supervision.detection.tools.polygon_zone.PolygonZone) to now accept a list of specific box anchors that must be in zone for a detection to be counted.

!!! failure "Deprecated"

    The `triggering_position ` parameter in `sv.PolygonZone` is deprecated and will be removed in `supervision-0.23.0`. Use `triggering_anchors` instead.

- Changed [#875](https://github.com/roboflow/supervision/pull/875): annotators adding support for Pillow images. All supervision Annotators can now accept an image as either a numpy array or a Pillow Image. They automatically detect its type, draw annotations, and return the output in the same format as the input.

- Fixed [#944](https://github.com/roboflow/supervision/pull/944): [`sv.DetectionsSmoother`](/0.19.0/detection/tools/smoother/#supervision.detection.tools.smoother.DetectionsSmoother) removing `tracking_id` from `sv.Detections`.

### 0.18.0 <small>January 25, 2024</small>

- Added [#720](https://github.com/roboflow/supervision/pull/720): [`sv.PercentageBarAnnotator`](/0.18.0/annotators/#percentagebarannotator) allowing to annotate images and videos with percentage values representing confidence or other custom property.

```python
>>> import supervision as sv

>>> image = ...
>>> detections = sv.Detections(...)

>>> percentage_bar_annotator = sv.PercentageBarAnnotator()
>>> annotated_frame = percentage_bar_annotator.annotate(
...     scene=image.copy(),
...     detections=detections
... )
```

- Added [#702](https://github.com/roboflow/supervision/pull/702): [`sv.RoundBoxAnnotator`](/0.18.0/annotators/#roundboxannotator) allowing to annotate images and videos with rounded corners bounding boxes.

- Added [#770](https://github.com/roboflow/supervision/pull/770): [`sv.OrientedBoxAnnotator`](/0.18.0/annotators/#orientedboxannotator) allowing to annotate images and videos with OBB (Oriented Bounding Boxes).

```python
import cv2
import supervision as sv
from ultralytics import YOLO

image = cv2.imread(<SOURCE_IMAGE_PATH>)
model = YOLO("yolov8n-obb.pt")

result = model(image)[0]
detections = sv.Detections.from_ultralytics(result)

oriented_box_annotator = sv.OrientedBoxAnnotator()
annotated_frame = oriented_box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

- Added [#696](https://github.com/roboflow/supervision/pull/696): [`sv.DetectionsSmoother`](/0.18.0/detection/tools/smoother/#detection-smoother) allowing for smoothing detections over multiple frames in video tracking.

- Added [#769](https://github.com/roboflow/supervision/pull/769): [`sv.ColorPalette.from_matplotlib`](/0.18.0/draw/color/#supervision.draw.color.ColorPalette.from_matplotlib) allowing users to create a `sv.ColorPalette` instance from a Matplotlib color palette.

```python
>>> import supervision as sv

>>> sv.ColorPalette.from_matplotlib('viridis', 5)
ColorPalette(colors=[Color(r=68, g=1, b=84), Color(r=59, g=82, b=139), ...])
```

- Changed [#770](https://github.com/roboflow/supervision/pull/770): [`sv.Detections.from_ultralytics`](/0.18.0/detection/core/#supervision.detection.core.Detections.from_ultralytics) adding support for OBB (Oriented Bounding Boxes).

- Changed [#735](https://github.com/roboflow/supervision/pull/735): [`sv.LineZone`](/0.18.0/detection/tools/line_zone/#linezone) to now accept a list of specific box anchors that must cross the line for a detection to be counted. This update marks a significant improvement from the previous requirement, where all four box corners were necessary. Users can now specify a single anchor, such as `sv.Position.BOTTOM_CENTER`, or any other combination of anchors defined as `List[sv.Position]`.

- Changed [#756](https://github.com/roboflow/supervision/pull/756): [`sv.Color`](/0.18.0/draw/color/#color)'s and [`sv.ColorPalette`](/0.18.0/draw/color/#colorpalette)'s method of accessing predefined colors, transitioning from a function-based approach (`sv.Color.red()`) to a more intuitive and conventional property-based method (`sv.Color.RED`).

!!! failure "Deprecated"

    `sv.ColorPalette.default()` is deprecated and will be removed in `supervision-0.22.0`. Use `sv.ColorPalette.DEFAULT` instead.

- Changed [#769](https://github.com/roboflow/supervision/pull/769): [`sv.ColorPalette.DEFAULT`](/0.18.0/draw/color/#colorpalette) value, giving users a more extensive set of annotation colors.

- Changed [#677](https://github.com/roboflow/supervision/pull/677): `sv.Detections.from_roboflow` to [`sv.Detections.from_inference`](/0.18.0/detection/core/#supervision.detection.core.Detections.from_inference) streamlining its functionality to be compatible with both the both [inference](https://github.com/roboflow/inference) pip package and the Robloflow [hosted API](https://docs.roboflow.com/deploy/hosted-api).

!!! failure "Deprecated"

    `Detections.from_roboflow()` is deprecated and will be removed in `supervision-0.22.0`. Use `Detections.from_inference` instead.

- Fixed [#735](https://github.com/roboflow/supervision/pull/735): [`sv.LineZone`](/0.18.0/detection/tools/line_zone/#linezone) functionality to accurately update the counter when an object crosses a line from any direction, including from the side. This enhancement enables more precise tracking and analytics, such as calculating individual in/out counts for each lane on the road.

### 0.17.0 <small>December 06, 2023</small>

- Added [#633](https://github.com/roboflow/supervision/pull/633): [`sv.PixelateAnnotator`](/0.17.0/annotators/#supervision.annotators.core.PixelateAnnotator) allowing to pixelate objects on images and videos.

- Added [#652](https://github.com/roboflow/supervision/pull/652): [`sv.TriangleAnnotator`](/0.17.0/annotators/#supervision.annotators.core.TriangleAnnotator) allowing to annotate images and videos with triangle markers.

- Added [#602](https://github.com/roboflow/supervision/pull/602): [`sv.PolygonAnnotator`](/0.17.0/annotators/#supervision.annotators.core.PolygonAnnotator) allowing to annotate images and videos with segmentation mask outline.

```python
>>> import supervision as sv

>>> image = ...
>>> detections = sv.Detections(...)

>>> polygon_annotator = sv.PolygonAnnotator()
>>> annotated_frame = polygon_annotator.annotate(
...     scene=image.copy(),
...     detections=detections
... )
```

- Added [#476](https://github.com/roboflow/supervision/pull/476): [`sv.assets`](/0.18.0/assets/) allowing download of video files that you can use in your demos.

```python
>>> from supervision.assets import download_assets, VideoAssets
>>> download_assets(VideoAssets.VEHICLES)
"vehicles.mp4"
```

- Added [#605](https://github.com/roboflow/supervision/pull/605): [`Position.CENTER_OF_MASS`](/0.17.0/geometry/core/#position) allowing to place labels in center of mass of segmentation masks.

- Added [#651](https://github.com/roboflow/supervision/pull/651): [`sv.scale_boxes`](/0.17.0/detection/utils/#supervision.detection.utils.scale_boxes) allowing to scale [`sv.Detections.xyxy`](/0.17.0/detection/core/#supervision.detection.core.Detections) values.

- Added [#637](https://github.com/roboflow/supervision/pull/637): [`sv.calculate_dynamic_text_scale`](/0.17.0/draw/utils/#supervision.draw.utils.calculate_dynamic_text_scale) and [`sv.calculate_dynamic_line_thickness`](/0.17.0/draw/utils/#supervision.draw.utils.calculate_dynamic_line_thickness) allowing text scale and line thickness to match image resolution.

- Added [#620](https://github.com/roboflow/supervision/pull/620): [`sv.Color.as_hex`](/0.17.0/draw/color/#supervision.draw.color.Color.as_hex) allowing to extract color value in HEX format.

- Added [#572](https://github.com/roboflow/supervision/pull/572): [`sv.Classifications.from_timm`](/0.17.0/classification/core/#supervision.classification.core.Classifications.from_timm) allowing to load classification result from [timm](https://huggingface.co/docs/hub/timm) models.

- Added [#478](https://github.com/roboflow/supervision/pull/478): [`sv.Classifications.from_clip`](/0.17.0/classification/core/#supervision.classification.core.Classifications.from_clip) allowing to load classification result from [clip](https://github.com/openai/clip) model.

- Added [#571](https://github.com/roboflow/supervision/pull/571): [`sv.Detections.from_azure_analyze_image`](/0.17.0/detection/core/#supervision.detection.core.Detections.from_azure_analyze_image) allowing to load detection results from [Azure Image Analysis](https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/concept-object-detection-40).

- Changed [#646](https://github.com/roboflow/supervision/pull/646): `sv.BoxMaskAnnotator` renaming it to [`sv.ColorAnnotator`](/0.17.0/annotators/#supervision.annotators.core.ColorAnnotator).

- Changed [#606](https://github.com/roboflow/supervision/pull/606): [`sv.MaskAnnotator`](/0.17.0/annotators/#supervision.annotators.core.MaskAnnotator) to make it **5x faster**.

- Fixed [#584](https://github.com/roboflow/supervision/pull/584): [`sv.DetectionDataset.from_yolo`](/0.17.0/datasets/#supervision.dataset.core.DetectionDataset.from_yolo) to ignore empty lines in annotation files.

- Fixed [#555](https://github.com/roboflow/supervision/pull/555): [`sv.BlurAnnotator`](/0.17.0/annotators/#supervision.annotators.core.BlurAnnotator) to trim negative coordinates before bluring detections.

- Fixed [#511](https://github.com/roboflow/supervision/pull/511): [`sv.TraceAnnotator`](/0.17.0/annotators/#supervision.annotators.core.TraceAnnotator) to respect trace position.

### 0.16.0 <small>October 19, 2023</small>

- Added [#422](https://github.com/roboflow/supervision/pull/422): [`sv.BoxMaskAnnotator`](/0.16.0/annotators/#supervision.annotators.core.BoxMaskAnnotator) allowing to annotate images and videos with mox masks.

- Added [#433](https://github.com/roboflow/supervision/pull/433): [`sv.HaloAnnotator`](/0.16.0/annotators/#supervision.annotators.core.HaloAnnotator) allowing to annotate images and videos with halo effect.

```python
>>> import supervision as sv

>>> image = ...
>>> detections = sv.Detections(...)

>>> halo_annotator = sv.HaloAnnotator()
>>> annotated_frame = halo_annotator.annotate(
...     scene=image.copy(),
...     detections=detections
... )
```

- Added [#466](https://github.com/roboflow/supervision/pull/466): [`sv.HeatMapAnnotator`](/0.16.0/annotators/#supervision.annotators.core.HeatMapAnnotator) allowing to annotate videos with heat maps.

- Added [#492](https://github.com/roboflow/supervision/pull/492): [`sv.DotAnnotator`](/0.16.0/annotators/#supervision.annotators.core.DotAnnotator) allowing to annotate images and videos with dots.

- Added [#449](https://github.com/roboflow/supervision/pull/449): [`sv.draw_image`](/0.16.0/draw/utils/#supervision.draw.utils.draw_image) allowing to draw an image onto a given scene with specified opacity and dimensions.

- Added [#280](https://github.com/roboflow/supervision/pull/280): [`sv.FPSMonitor`](/0.16.0/utils/video/#supervision.utils.video.FPSMonitor) for monitoring frames per second (FPS) to benchmark latency.

- Added [#454](https://github.com/roboflow/supervision/pull/454): 🤗 Hugging Face Annotators [space](https://huggingface.co/spaces/Roboflow/Annotators).

- Changed [#482](https://github.com/roboflow/supervision/pull/482): [`sv.LineZone.trigger`](/0.16.0/detection/tools/line_zone/#supervision.detection.line_counter.LineZone.trigger) now return `Tuple[np.ndarray, np.ndarray]`. The first array indicates which detections have crossed the line from outside to inside. The second array indicates which detections have crossed the line from inside to outside.

- Changed [#465](https://github.com/roboflow/supervision/pull/465): Annotator argument name from `color_map: str` to `color_lookup: ColorLookup` enum to increase type safety.

- Changed [#426](https://github.com/roboflow/supervision/pull/426): [`sv.MaskAnnotator`](/0.16.0/annotators/#supervision.annotators.core.MaskAnnotator) allowing 2x faster annotation.

- Fixed [#477](https://github.com/roboflow/supervision/pull/477): Poetry env definition allowing proper local installation.

- Fixed [#430](https://github.com/roboflow/supervision/pull/430): [`sv.ByteTrack`](/0.16.0/trackers/#supervision.tracker.byte_tracker.core.ByteTrack) to return `np.array([], dtype=int)` when `svDetections` is empty.

!!! failure "Deprecated"

    `sv.Detections.from_yolov8` and `sv.Classifications.from_yolov8` as those are now replaced by [`sv.Detections.from_ultralytics`](/0.16.0/detection/core/#supervision.detection.core.Detections.from_ultralytics) and [`sv.Classifications.from_ultralytics`](/0.16.0/classification/core/#supervision.classification.core.Classifications.from_ultralytics).

### 0.15.0 <small>October 5, 2023</small>

- Added [#170](https://github.com/roboflow/supervision/pull/170): [`sv.BoundingBoxAnnotator`](/0.15.0/annotators/#supervision.annotators.core.BoundingBoxAnnotator) allowing to annotate images and videos with bounding boxes.

- Added [#170](https://github.com/roboflow/supervision/pull/170): [`sv.BoxCornerAnnotator `](/0.15.0/annotators/#supervision.annotators.core.BoxCornerAnnotator) allowing to annotate images and videos with just bounding box corners.

- Added [#170](https://github.com/roboflow/supervision/pull/170): [`sv.MaskAnnotator`](/0.15.0/annotators/#supervision.annotators.core.MaskAnnotator) allowing to annotate images and videos with segmentation masks.

- Added [#170](https://github.com/roboflow/supervision/pull/170): [`sv.EllipseAnnotator`](/0.15.0/annotators/#supervision.annotators.core.EllipseAnnotator) allowing to annotate images and videos with ellipses (sports game style).

- Added [#386](https://github.com/roboflow/supervision/pull/386): [`sv.CircleAnnotator`](/0.15.0/annotators/#supervision.annotators.core.CircleAnnotator) allowing to annotate images and videos with circles.

- Added [#354](https://github.com/roboflow/supervision/pull/354): [`sv.TraceAnnotator`](/0.15.0/annotators/#supervision.annotators.core.TraceAnnotator) allowing to draw path of moving objects on videos.

- Added [#405](https://github.com/roboflow/supervision/pull/405): [`sv.BlurAnnotator`](/0.15.0/annotators/#supervision.annotators.core.BlurAnnotator) allowing to blur objects on images and videos.

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

- Added [#354](https://github.com/roboflow/supervision/pull/354): Supervision usage [example](https://github.com/roboflow/supervision/tree/develop/examples/traffic_analysis). You can now learn how to perform traffic flow analysis with Supervision.

- Changed [#399](https://github.com/roboflow/supervision/pull/399): [`sv.Detections.from_roboflow`](/0.15.0/detection/core/#supervision.detection.core.Detections.from_roboflow) now does not require `class_list` to be specified. The `class_id` value can be extracted directly from the [inference](https://github.com/roboflow/inference) response.

- Changed [#381](https://github.com/roboflow/supervision/pull/381): [`sv.VideoSink`](/0.15.0/utils/video/#videosink) now allows to customize the output codec.

- Changed [#361](https://github.com/roboflow/supervision/pull/361): [`sv.InferenceSlicer`](/0.15.0/detection/tools/inference_slicer/#supervision.detection.tools.inference_slicer.InferenceSlicer) can now operate in multithreading mode.

- Fixed [#348](https://github.com/roboflow/supervision/pull/348): [`sv.Detections.from_deepsparse`](/0.15.0/detection/core/#supervision.detection.core.Detections.from_deepsparse) to allow processing empty [deepsparse](https://github.com/neuralmagic/deepsparse) result object.

### 0.14.0 <small>August 31, 2023</small>

- Added [#282](https://github.com/roboflow/supervision/pull/282): support for SAHI inference technique with [`sv.InferenceSlicer`](/0.14.0/detection/tools/inference_slicer).

```python
>>> import cv2
>>> import supervision as sv
>>> from ultralytics import YOLO

>>> image = cv2.imread(SOURCE_IMAGE_PATH)
>>> model = YOLO(...)

>>> def callback(image_slice: np.ndarray) -> sv.Detections:
...     result = model(image_slice)[0]
...     return sv.Detections.from_ultralytics(result)

>>> slicer = sv.InferenceSlicer(callback = callback)

>>> detections = slicer(image)
```

- Added [#297](https://github.com/roboflow/supervision/pull/297): [`Detections.from_deepsparse`](/0.14.0/detection/core/#supervision.detection.core.Detections.from_deepsparse) to enable seamless integration with [DeepSparse](https://github.com/neuralmagic/deepsparse) framework.

- Added [#281](https://github.com/roboflow/supervision/pull/281): [`sv.Classifications.from_ultralytics`](/0.14.0/classification/core/#supervision.classification.core.Classifications.from_ultralytics) to enable seamless integration with [Ultralytics](https://github.com/ultralytics/ultralytics) framework. This will enable you to use supervision with all [models](https://docs.ultralytics.com/models/) that Ultralytics supports.

!!! failure "Deprecated"

    [sv.Detections.from_yolov8](/0.14.0/detection/core/#supervision.detection.core.Detections.from_yolov8) and [sv.Classifications.from_yolov8](/0.14.0/classification/core/#supervision.classification.core.Classifications.from_yolov8) are now deprecated and will be removed with `supervision-0.16.0` release.

- Added [#341](https://github.com/roboflow/supervision/pull/341): First supervision usage example script showing how to detect and track objects on video using YOLOv8 + Supervision.

- Changed [#296](https://github.com/roboflow/supervision/pull/296): [`sv.ClassificationDataset`](/0.14.0/dataset/core/#supervision.dataset.core.ClassificationDataset) and [`sv.DetectionDataset`](/0.14.0/dataset/core/#supervision.dataset.core.DetectionDataset) now use image path (not image name) as dataset keys.

- Fixed [#300](https://github.com/roboflow/supervision/pull/300): [`Detections.from_roboflow`](/0.14.0/detection/core/#supervision.detection.core.Detections.from_roboflow) to filter out polygons with less than 3 points.

### 0.13.0 <small>August 8, 2023</small>

- Added [#236](https://github.com/roboflow/supervision/pull/236): support for mean average precision (mAP) for object detection models with [`sv.MeanAveragePrecision`](/0.13.0/metrics/detection/#meanaverageprecision).

```python
>>> import supervision as sv
>>> from ultralytics import YOLO

>>> dataset = sv.DetectionDataset.from_yolo(...)

>>> model = YOLO(...)
>>> def callback(image: np.ndarray) -> sv.Detections:
...     result = model(image)[0]
...     return sv.Detections.from_yolov8(result)

>>> mean_average_precision = sv.MeanAveragePrecision.benchmark(
...     dataset = dataset,
...     callback = callback
... )

>>> mean_average_precision.map50_95
0.433
```

- Added [#256](https://github.com/roboflow/supervision/pull/256): support for ByteTrack for object tracking with [`sv.ByteTrack`](/0.13.0/tracker/core/#bytetrack).

- Added [#222](https://github.com/roboflow/supervision/pull/222): [`sv.Detections.from_ultralytics`](/0.13.0/detection/core/#supervision.detection.core.Detections.from_ultralytics) to enable seamless integration with [Ultralytics](https://github.com/ultralytics/ultralytics) framework. This will enable you to use `supervision` with all [models](https://docs.ultralytics.com/models/) that Ultralytics supports.

!!! failure "Deprecated"

    [`sv.Detections.from_yolov8`](/0.13.0/detection/core/#supervision.detection.core.Detections.from_yolov8) is now deprecated and will be removed with `supervision-0.15.0` release.

- Added [#191](https://github.com/roboflow/supervision/pull/191): [`sv.Detections.from_paddledet`](/0.13.0/detection/core/#supervision.detection.core.Detections.from_paddledet) to enable seamless integration with [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) framework.

- Added [#245](https://github.com/roboflow/supervision/pull/245): support for loading PASCAL VOC segmentation datasets with [`sv.DetectionDataset.`](/0.13.0/dataset/core/#supervision.dataset.core.DetectionDataset.from_pascal_voc).

### 0.12.0 <small>July 24, 2023</small>

!!! failure "Python 3.7. Support Terminated"

    With the `supervision-0.12.0` release, we are terminating official support for Python 3.7.

- Added [#177](https://github.com/roboflow/supervision/pull/177): initial support for object detection model benchmarking with [`sv.ConfusionMatrix`](/0.12.0/metrics/detection/#confusionmatrix).

```python
>>> import supervision as sv
>>> from ultralytics import YOLO

>>> dataset = sv.DetectionDataset.from_yolo(...)

>>> model = YOLO(...)
>>> def callback(image: np.ndarray) -> sv.Detections:
...     result = model(image)[0]
...     return sv.Detections.from_yolov8(result)

>>> confusion_matrix = sv.ConfusionMatrix.benchmark(
...     dataset = dataset,
...     callback = callback
... )

>>> confusion_matrix.matrix
array([
    [0., 0., 0., 0.],
    [0., 1., 0., 1.],
    [0., 1., 1., 0.],
    [1., 1., 0., 0.]
])
```

- Added [#173](https://github.com/roboflow/supervision/pull/173): [`Detections.from_mmdetection`](/0.12.0/detection/core/#supervision.detection.core.Detections.from_mmdetection) to enable seamless integration with [MMDetection](https://github.com/open-mmlab/mmdetection) framework.

- Added [#130](https://github.com/roboflow/supervision/issues/130): ability to [install](https://supervision.roboflow.com/) package in `headless` or `desktop` mode.

- Changed [#180](https://github.com/roboflow/supervision/pull/180): packing method from `setup.py` to `pyproject.toml`.

- Fixed [#188](https://github.com/roboflow/supervision/issues/188): [`sv.DetectionDataset.from_cooc`](/0.12.0/dataset/core/#supervision.dataset.core.DetectionDataset.from_coco) can't be loaded when there are images without annotations.

- Fixed [#226](https://github.com/roboflow/supervision/issues/226): [`sv.DetectionDataset.from_yolo`](/0.12.0/dataset/core/#supervision.dataset.core.DetectionDataset.from_yolo) can't load background instances.

### 0.11.1 <small>June 29, 2023</small>

- Fix [#165](https://github.com/roboflow/supervision/pull/165): [`as_folder_structure`](/0.11.1/dataset/core/#supervision.dataset.core.ClassificationDataset.as_folder_structure) fails to save [`sv.ClassificationDataset`](/0.11.1/dataset/core/#classificationdataset) when it is result of inference.

### 0.11.0 <small>June 28, 2023</small>

- Added [#150](https://github.com/roboflow/supervision/pull/150): ability to load and save [`sv.DetectionDataset`](/0.11.0/dataset/core/#detectiondataset) in COCO format using [`as_coco`](/0.11.0/dataset/core/#supervision.dataset.core.DetectionDataset.as_coco) and [`from_coco`](/0.11.0/dataset/core/#supervision.dataset.core.DetectionDataset.from_coco) methods.

```python
>>> import supervision as sv

>>> ds = sv.DetectionDataset.from_coco(
...     images_directory_path='...',
...     annotations_path='...'
... )

>>> ds.as_coco(
...     images_directory_path='...',
...     annotations_path='...'
... )
```

- Added [#158](https://github.com/roboflow/supervision/pull/158): ability to merge multiple [`sv.DetectionDataset`](/0.11.0/dataset/core/#detectiondataset) together using [`merge`](/0.11.0/dataset/core/#supervision.dataset.core.DetectionDataset.merge) method.

```python
>>> import supervision as sv

>>> ds_1 = sv.DetectionDataset(...)
>>> len(ds_1)
100
>>> ds_1.classes
['dog', 'person']

>>> ds_2 = sv.DetectionDataset(...)
>>> len(ds_2)
200
>>> ds_2.classes
['cat']

>>> ds_merged = sv.DetectionDataset.merge([ds_1, ds_2])
>>> len(ds_merged)
300
>>> ds_merged.classes
['cat', 'dog', 'person']
```

- Added [#162](https://github.com/roboflow/supervision/pull/162): additional `start` and `end` arguments to [`sv.get_video_frames_generator`](/0.11.0/utils/video/#get_video_frames_generator) allowing to generate frames only for a selected part of the video.

- Fix [#157](https://github.com/roboflow/supervision/pull/157): incorrect loading of YOLO dataset class names from `data.yaml`.

### 0.10.0 <small>June 14, 2023</small>

- Added [#125](https://github.com/roboflow/supervision/pull/125): ability to load and save [`sv.ClassificationDataset`](/0.10.0/dataset/core/#classificationdataset) in a folder structure format.

```python
>>> import supervision as sv

>>> cs = sv.ClassificationDataset.from_folder_structure(
...     root_directory_path='...'
... )

>>> cs.as_folder_structure(
...     root_directory_path='...'
... )
```

- Added [#125](https://github.com/roboflow/supervision/pull/125): support for [`sv.ClassificationDataset.split`](/0.10.0/dataset/core/#supervision.dataset.core.ClassificationDataset.split) allowing to divide `sv.ClassificationDataset` into two parts.

- Added [#110](https://github.com/roboflow/supervision/pull/110): ability to extract masks from Roboflow API results using [`sv.Detections.from_roboflow`](/0.10.0/detection/core/#supervision.detection.core.Detections.from_roboflow).

- Added [commit hash](https://github.com/roboflow/supervision/commit/d000292eb2f2342544e0947b65528082e60fb8d6): Supervision Quickstart [notebook](https://colab.research.google.com/github/roboflow/supervision/blob/main/demo.ipynb) where you can learn more about Detection, Dataset and Video APIs.

- Changed [#135](https://github.com/roboflow/supervision/pull/135): `sv.get_video_frames_generator` documentation to better describe actual behavior.

### 0.9.0 <small>June 7, 2023</small>

- Added [#118](https://github.com/roboflow/supervision/pull/118): ability to select [`sv.Detections`](/0.9.0/detection/core/#supervision.detection.core.Detections.__getitem__) by index, list of indexes or slice. Here is an example illustrating the new selection methods.

```python
>>> import supervision as sv

>>> detections = sv.Detections(...)
>>> len(detections[0])
1
>>> len(detections[[0, 1]])
2
>>> len(detections[0:2])
2
```

- Added [#101](https://github.com/roboflow/supervision/pull/101): ability to extract masks from YOLOv8 result using [`sv.Detections.from_yolov8`](/0.8.0/detection/core/#supervision.detection.core.Detections.from_yolov8). Here is an example illustrating how to extract boolean masks from the result of the YOLOv8 model inference.

- Added [#122](https://github.com/roboflow/supervision/pull/122): ability to crop image using [`sv.crop`](/latest/utils/image/#crop). Here is an example showing how to get a separate crop for each detection in `sv.Detections`.

- Added [#120](https://github.com/roboflow/supervision/pull/120): ability to conveniently save multiple images into directory using [`sv.ImageSink`](/0.9.0/utils/image/#imagesink). Here is an example showing how to save every tenth video frame as a separate image.

```python
>>> import supervision as sv

>>> with sv.ImageSink(target_dir_path='target/directory/path') as sink:
...     for image in sv.get_video_frames_generator(source_path='source_video.mp4', stride=10):
...         sink.save_image(image=image)
```

- Fixed [#106](https://github.com/roboflow/supervision/issues/106): inconvenient handling of [`sv.PolygonZone`](/0.8.0/supervision/detection/tools/polygon_zone/#polygonzone) coordinates. Now `sv.PolygonZone` accepts coordinates in the form of `[[x1, y1], [x2, y2], ...]` that can be both integers and floats.

### 0.8.0 <small>May 17, 2023</small>

- Added [#100](https://github.com/roboflow/supervision/pull/100): support for dataset inheritance. The current `Dataset` got renamed to `DetectionDataset`. Now [`DetectionDataset`](/0.8.0/supervision/dataset/core/#detectiondataset) inherits from `BaseDataset`. This change was made to enforce the future consistency of APIs of different types of computer vision datasets.
- Added [#100](https://github.com/roboflow/supervision/pull/100): ability to save datasets in YOLO format using [`DetectionDataset.as_yolo`](/0.8.0/dataset/core/#supervision.dataset.core.DetectionDataset.as_yolo).

```python
>>> import roboflow
>>> from roboflow import Roboflow
>>> import supervision as sv

>>> roboflow.login()

>>> rf = Roboflow()

>>> project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
>>> dataset = project.version(PROJECT_VERSION).download("yolov5")

>>> ds = sv.DetectionDataset.from_yolo(
...     images_directory_path=f"{dataset.location}/train/images",
...     annotations_directory_path=f"{dataset.location}/train/labels",
...     data_yaml_path=f"{dataset.location}/data.yaml"
... )

>>> ds.classes
['dog', 'person']
```

- Added [#102](https://github.com/roboflow/supervision/pull/103): support for [`DetectionDataset.split`](/0.8.0/dataset/core/#supervision.dataset.core.DetectionDataset.split) allowing to divide `DetectionDataset` into two parts.

```python
>>> import supervision as sv

>>> ds = sv.DetectionDataset(...)
>>> train_ds, test_ds = ds.split(split_ratio=0.7, random_state=42, shuffle=True)

>>> len(train_ds), len(test_ds)
(700, 300)
```

- Changed [#100](https://github.com/roboflow/supervision/pull/100): default value of `approximation_percentage` parameter from `0.75` to `0.0` in `DetectionDataset.as_yolo` and `DetectionDataset.as_pascal_voc`.

### 0.7.0 <small>May 11, 2023</small>

- Added [#91](https://github.com/roboflow/supervision/pull/91): `Detections.from_yolo_nas` to enable seamless integration with [YOLO-NAS](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md) model.
- Added [#86](https://github.com/roboflow/supervision/pull/86): ability to load datasets in YOLO format using `Dataset.from_yolo`.
- Added [#84](https://github.com/roboflow/supervision/pull/84): `Detections.merge` to merge multiple `Detections` objects together.
- Fixed [#81](https://github.com/roboflow/supervision/pull/81): `LineZoneAnnotator.annotate` does not return annotated frame.
- Changed [#44](https://github.com/roboflow/supervision/pull/44): `LineZoneAnnotator.annotate` to allow for custom text for the in and out tags.

### 0.6.0 <small>April 19, 2023</small>

- Added [#71](https://github.com/roboflow/supervision/pull/71): initial `Dataset` support and ability to save `Detections` in Pascal VOC XML format.
- Added [#71](https://github.com/roboflow/supervision/pull/71): new `mask_to_polygons`, `filter_polygons_by_area`, `polygon_to_xyxy` and `approximate_polygon` utilities.
- Added [#72](https://github.com/roboflow/supervision/pull/72): ability to load Pascal VOC XML **object detections** dataset as `Dataset`.
- Changed [#70](https://github.com/roboflow/supervision/pull/70): order of `Detections` attributes to make it consistent with order of objects in `__iter__` tuple.
- Changed [#71](https://github.com/roboflow/supervision/pull/71): `generate_2d_mask` to `polygon_to_mask`.

### 0.5.2 <small>April 13, 2023</small>

- Fixed [#63](https://github.com/roboflow/supervision/pull/63): `LineZone.trigger` function expects 4 values instead of 5.

### 0.5.1 <small>April 12, 2023</small>

- Fixed `Detections.__getitem__` method did not return mask for selected item.
- Fixed `Detections.area` crashed for mask detections.

### 0.5.0 <small>April 10, 2023</small>

- Added [#58](https://github.com/roboflow/supervision/pull/58): `Detections.mask` to enable segmentation support.
- Added [#58](https://github.com/roboflow/supervision/pull/58): `MaskAnnotator` to allow easy `Detections.mask` annotation.
- Added [#58](https://github.com/roboflow/supervision/pull/58): `Detections.from_sam` to enable native Segment Anything Model (SAM) support.
- Changed [#58](https://github.com/roboflow/supervision/pull/58): `Detections.area` behaviour to work not only with boxes but also with masks.

### 0.4.0 <small>April 5, 2023</small>

- Added [#46](https://github.com/roboflow/supervision/discussions/48): `Detections.empty` to allow easy creation of empty `Detections` objects.
- Added [#56](https://github.com/roboflow/supervision/pull/56): `Detections.from_roboflow` to allow easy creation of `Detections` objects from Roboflow API inference results.
- Added [#56](https://github.com/roboflow/supervision/pull/56): `plot_images_grid` to allow easy plotting of multiple images on single plot.
- Added [#56](https://github.com/roboflow/supervision/pull/56): initial support for Pascal VOC XML format with `detections_to_voc_xml` method.
- Changed [#56](https://github.com/roboflow/supervision/pull/56): `show_frame_in_notebook` refactored and renamed to `plot_image`.

### 0.3.2 <small>March 23, 2023</small>

- Changed [#50](https://github.com/roboflow/supervision/issues/50): Allow `Detections.class_id` to be `None`.

### 0.3.1 <small>March 6, 2023</small>

- Fixed [#41](https://github.com/roboflow/supervision/issues/41): `PolygonZone` throws an exception when the object touches the bottom edge of the image.
- Fixed [#42](https://github.com/roboflow/supervision/issues/42): `Detections.wth_nms` method throws an exception when `Detections` is empty.
- Changed [#36](https://github.com/roboflow/supervision/pull/36): `Detections.wth_nms` support class agnostic and non-class agnostic case.

### 0.3.0 <small>March 6, 2023</small>

- Changed: Allow `Detections.confidence` to be `None`.
- Added: `Detections.from_transformers` and `Detections.from_detectron2` to enable seamless integration with Transformers and Detectron2 models.
- Added: `Detections.area` to dynamically calculate bounding box area.
- Added: `Detections.wth_nms` to filter out double detections with NMS. Initial - only class agnostic - implementation.

### 0.2.0 <small>February 2, 2023</small>

- Added: Advanced `Detections` filtering with pandas-like API.
- Added: `Detections.from_yolov5` and `Detections.from_yolov8` to enable seamless integration with YOLOv5 and YOLOv8 models.

### 0.1.0 <small>January 19, 2023</small>

Say hello to Supervision 👋
