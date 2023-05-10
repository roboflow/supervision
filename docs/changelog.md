### 0.7.0 <small>May 11, 2023</small>

- Added [[#91](https://github.com/roboflow/supervision/pull/91)]: `Detections.from_yolo_nas` to enable seamless integration with [YOLO-NAS](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md) model.
- Added [[#86](https://github.com/roboflow/supervision/pull/86)]: ability to load datasets in YOLO format using `Dataset.from_yolo`. 
- Added [[#84](https://github.com/roboflow/supervision/pull/84)]: `Detections.merge` to merge multiple `Detections` objects together.
- Fixed [[#81](https://github.com/roboflow/supervision/pull/81)]: `LineZoneAnnotator.annotate` does not return annotated frame.
- Changed [[#44](https://github.com/roboflow/supervision/pull/44)]: `LineZoneAnnotator.annotate` to allow for custom text for the in and out tags.

### 0.6.0 <small>April 19, 2023</small>

- Added [[#71](https://github.com/roboflow/supervision/pull/71)]: initial `Dataset` support and ability to save `Detections` in Pascal VOC XML format. 
- Added [[#71](https://github.com/roboflow/supervision/pull/71)]: new `mask_to_polygons`, `filter_polygons_by_area`, `polygon_to_xyxy` and `approximate_polygon` utilities.
- Added [[#72](https://github.com/roboflow/supervision/pull/72)]: ability to load Pascal VOC XML **object detections** dataset as `Dataset`.
- Changed [[#70](https://github.com/roboflow/supervision/pull/70)]: order of `Detections` attributes to make it consistent with order of objects in `__iter__` tuple.
- Changed [[#71](https://github.com/roboflow/supervision/pull/71)]: `generate_2d_mask` to `polygon_to_mask`.

### 0.5.2 <small>April 13, 2023</small>

- Fixed [[#63](https://github.com/roboflow/supervision/pull/63)]: `LineZone.trigger` function expects 4 values instead of 5.

### 0.5.1 <small>April 12, 2023</small>

- Fixed `Detections.__getitem__` method did not return mask for selected item.
- Fixed `Detections.area` crashed for mask detections.

### 0.5.0 <small>April 10, 2023</small>

- Added [[#58](https://github.com/roboflow/supervision/pull/58)]: `Detections.mask` to enable segmentation support.
- Added [[#58](https://github.com/roboflow/supervision/pull/58)]: `MaskAnnotator` to allow easy `Detections.mask` annotation.
- Added [[#58](https://github.com/roboflow/supervision/pull/58)]: `Detections.from_sam` to enable native Segment Anything Model (SAM) support.
- Changed [[#58](https://github.com/roboflow/supervision/pull/58)]: `Detections.area` behaviour to work not only with boxes but also with masks.

### 0.4.0 <small>April 5, 2023</small> 

- Added [[#46](https://github.com/roboflow/supervision/discussions/48)]: `Detections.empty` to allow easy creation of empty `Detections` objects.
- Added [[#56](https://github.com/roboflow/supervision/pull/56)]: `Detections.from_roboflow` to allow easy creation of `Detections` objects from Roboflow API inference results.
- Added [[#56](https://github.com/roboflow/supervision/pull/56)]: `plot_images_grid` to allow easy plotting of multiple images on single plot.
- Added [[#56](https://github.com/roboflow/supervision/pull/56)]: initial support for Pascal VOC XML format with `detections_to_voc_xml` method.
- Changed [[#56](https://github.com/roboflow/supervision/pull/56)]: `show_frame_in_notebook` refactored and renamed to `plot_image`.

### 0.3.2 <small>March 23, 2023</small> 

- Changed [[#50](https://github.com/roboflow/supervision/issues/50)]: Allow `Detections.class_id` to be `None`. 

### 0.3.1 <small>March 6, 2023</small> 

- Fixed [[#41](https://github.com/roboflow/supervision/issues/41)]: `PolygonZone` throws an exception when the object touches the bottom edge of the image.
- Fixed [[#42](https://github.com/roboflow/supervision/issues/42)]: `Detections.wth_nms` method throws an exception when `Detections` is empty.
- Changed [[#36](https://github.com/roboflow/supervision/pull/36)]: `Detections.wth_nms` support class agnostic and non-class agnostic case.

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
