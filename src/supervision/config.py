CLASS_NAME_DATA_FIELD: str = "class_name"
COCO_RAW_SEGMENTATION: str = "coco_raw_segmentation"
#: Key for per-detection area metadata in ``Detections.data``.
AREA_DATA_FIELD: str = "area"
#: Key for per-object axis-aligned boxes in ``KeyPoints.data``.
#:
#: Value layout: ``np.ndarray`` of shape ``(N, 4)`` in
#: ``(x_min, y_min, x_max, y_max)`` format, one box per skeleton. Read by
#: ``KeyPointMeanAveragePrecision``.
XYXY_DATA_FIELD: str = "xyxy"
#: Key for per-target COCO crowd flags in ``Detections.data`` or ``KeyPoints.data``.
#:
#: Value layout: ``np.ndarray`` of shape ``(N,)``; a non-zero value marks a
#: crowd region, as COCO ``iscrowd=1``. Read by the mAP metrics.
ISCROWD_DATA_FIELD: str = "iscrowd"
#: Key for the MediaPipe hand-handedness score in ``KeyPoints.data``.
#:
#: Value layout: ``np.ndarray`` of shape ``(N,)``, dtype ``float32``, holding the
#: top-1 handedness classification score of each detected hand as reported by
#: :meth:`~supervision.key_points.core.KeyPoints.from_mediapipe`. MediaPipe floors
#: this score at ``0.5`` — it rates confidence in the ``Left``/``Right`` label, not
#: the quality of the detection, so it is kept out of ``detection_confidence``.
HANDEDNESS_SCORE_DATA_FIELD: str = "handedness_score"
#: Key for the source image in ``Detections.metadata``.
#:
#: An RF-DETR / ``inference``-package convention rather than a field supervision
#: itself populates: model connectors from those packages attach the image the
#: predictions were produced from under this key.
#: :class:`~supervision.detection.tools.inference_slicer.InferenceSlicer` drops it
#: while merging slices (each slice carries a different tile) and restores the full
#: input image afterwards.
#: The stored value is a reference to the caller's image, not a copy — mutating
#: ``metadata[SOURCE_IMAGE_METADATA_FIELD]`` mutates the original array.
SOURCE_IMAGE_METADATA_FIELD: str = "source_image"
#: Key for oriented bounding-box corner coordinates in ``Detections.data``.
#:
#: Value layout: ``np.ndarray`` of shape ``(N, 4, 2)``, dtype ``float32``, pixel
#: coordinates ordered as ``[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]`` per
#: detection where the four points are the corners of the oriented box.
#: Used by :func:`~supervision.dataset.formats.yolo.detections_to_yolo_annotations`
#: (``is_obb=True``) and
#: :func:`~supervision.dataset.formats.yolo.yolo_annotations_to_detections`
#: (``is_obb=True``).
#: Also triggers sequential mode in ``InferenceSlicer`` when present.
ORIENTED_BOX_COORDINATES: str = "xyxyxyxy"
#: Key for per-detection metric depth in ``Detections.data``.
#:
#: Value layout: ``np.ndarray`` of shape ``(N,)``, dtype ``float32``, the median depth
#: in metres of the valid depth pixels inside each detection's mask (or box, when the
#: detections carry no masks), ``NaN`` where the region holds no depth. Written by
#: :meth:`~supervision.depth.core.DepthMap.measure_detections`.
DEPTH_M_DATA_FIELD: str = "depth_m"
#: Key for per-detection stereo disparity in ``Detections.data``.
#:
#: Same layout as ``DEPTH_M_DATA_FIELD``, in pixels of the depth map.
DISPARITY_PX_DATA_FIELD: str = "disparity_px"
#: Key for per-detection relative inverse depth in ``Detections.data``.
#:
#: Same layout as ``DEPTH_M_DATA_FIELD``, unitless; larger is nearer. Monocular models
#: such as Depth Anything produce it, and it has no metric scale.
RELATIVE_INVERSE_DATA_FIELD: str = "relative_inverse"
