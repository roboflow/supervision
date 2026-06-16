CLASS_NAME_DATA_FIELD: str = "class_name"
COCO_RAW_SEGMENTATION: str = "coco_raw_segmentation"
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
