import argparse
from collections import defaultdict
from typing import Generator, Literal

import cv2
import numpy as np
from ultralytics import YOLO

import supervision as sv

PRICES = {
    "banana": 1.0,
    "orange": 1.5,
}
BASKET = defaultdict(int)


MODEL = YOLO("yolov8l.pt")
TRACKER = sv.ByteTrack()

LABEL_ANNOTATOR = sv.LabelAnnotator(
    text_scale=1, text_thickness=4, text_color=sv.Color.BLACK)
BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=4)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-Service Checkout using YOLOv9 and Supervision"
    )
    return parser.parse_args()


def webcam_frame_generator() -> Generator[np.ndarray, None, None]:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        yield frame
    cap.release()


def update_basket(detections: sv.Detections, operation: Literal["add", "remove"]) -> None:
    for class_id in detections.class_id:
        class_name = MODEL.model.names[class_id]
        if operation == "add":
            BASKET[class_name] += 1
        elif operation == "remove":
            BASKET[class_name] = max(0, BASKET[class_name] - 1)


if __name__ == "__main__":
    args = parse_arguments()
    frame_generator = webcam_frame_generator()
    frame = next(frame_generator)
    frame_height, frame_width, _ = frame.shape

    start = sv.Point(frame_width // 2, frame_height)
    end = sv.Point(frame_width // 2, 0)
    line_zone = sv.LineZone(
        start=start, end=end, triggering_anchors=[sv.Position.CENTER])

    for frame in frame_generator:

        result = MODEL(frame, device="mps")[0]
        detections = sv.Detections.from_ultralytics(result)
        masks = (detections.class_id != 0) & (detections.class_id != 60)
        detections = detections[masks]
        detections = TRACKER.update_with_detections(detections=detections)

        crossed_in, crossed_out = line_zone.trigger(detections=detections)
        update_basket(detections[crossed_out], "add")
        update_basket(detections[crossed_in], "remove")

        labels = [
            f"#{tracker_id} {MODEL.model.names[class_id]}"
            for tracker_id, class_id
            in zip(detections.tracker_id, detections.class_id)
        ]

        annotated_frame = frame.copy()
        annotated_frame = sv.draw_line(
            scene=annotated_frame, start=start, end=end, color=sv.Color.WHITE)
        annotated_frame = BOUNDING_BOX_ANNOTATOR.annotate(
            scene=annotated_frame, detections=detections)
        annotated_frame = LABEL_ANNOTATOR.annotate(
            scene=annotated_frame, detections=detections, labels=labels)

        basket_value = sum(BASKET[product] * PRICES[product] for product in BASKET)
        text = f"${basket_value:0.1f}"
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=text,
            text_scale=4,
            text_thickness=4,
            text_anchor=sv.Point(frame_width // 2, frame_height - 100),
            text_color=sv.Color.WHITE,
            background_color=sv.Color.BLACK,
        )

        cv2.imshow('Self-Service Checkout Feed', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
