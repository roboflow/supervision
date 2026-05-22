CLASS_NAME_DATA_FIELD: str = "class_name"
# Used by move_detections (coordinate transform) and InferenceSlicer (OBB thread-safety
# detection). Any code that sets this key in Detections.data also affects slicer
# scheduling: InferenceSlicer switches to sequential mode when this key is present.
ORIENTED_BOX_COORDINATES: str = "xyxyxyxy"
