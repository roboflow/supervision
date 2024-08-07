import cv2
from ultralytics import YOLO

import supervision as sv

image = cv2.imread("basketball.jpg")
model = YOLO("yolov8s-pose.pt")
result = model(image)[0]
key_points = sv.KeyPoints.from_ultralytics(result)

COLORS = [
    "#FF6347",
    "#FF6347",
    "#FF6347",
    "#FF6347",
    "#FF6347",
    "#FF1493",
    "#00FF00",
    "#FF1493",
    "#00FF00",
    "#FF1493",
    "#00FF00",
    "#FFD700",
    "#00BFFF",
    "#FFD700",
    "#00BFFF",
    "#FFD700",
    "#00BFFF",
]

COLORS = [sv.Color.from_hex(color_hex=c) for c in COLORS]

vertex_label_annotator = sv.VertexLabelAnnotator(
    color=COLORS, text_color=sv.ColorPalette.ROBOFLOW, border_radius=5
)
annotated_frame = vertex_label_annotator.annotate(
    scene=image.copy(), key_points=key_points
)

sv.plot_image(annotated_frame)
