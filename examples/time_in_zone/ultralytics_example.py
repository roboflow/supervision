from typing import Generator

import cv2
import numpy as np
from ultralytics import YOLO

import supervision as sv

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])

POLYGONS = [
    np.array(
        [
            [684, 105],
            [845, 110],
            [888, 188],
            [925, 349],
            [981, 511],
            [944, 620],
            [728, 640],
        ]
    ),
    np.array(
        [
            [964, 101],
            [1124, 332],
            [1205, 480],
            [1146, 588],
            [1277, 562],
            [1280, 380],
            [1126, 96],
        ]
    ),
    np.array(
        [
            [402, 137],
            [499, 138],
            [471, 209],
            [465, 378],
            [434, 540],
            [478, 648],
            [299, 641],
            [226, 539],
            [279, 387],
        ]
    ),
]


def rtsp_stream_frames_generator(rtsp_url: str) -> Generator[np.ndarray, None, None]:
    """
    Generator function to yield frames from an RTSP stream.

    Args:
        rtsp_url (str): URL of the RTSP video stream.

    Yields:
        np.ndarray: The next frame from the video stream.
    """
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise Exception("Error: Could not open video stream.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream or error reading frame.")
                break
            yield frame
    finally:
        cap.release()


def main(rtsp_url: str, weights: str, device: str, confidence_threshold: float) -> None:
    model = YOLO(weights)
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_color=sv.Color.from_hex("#000000"))

    for frame in rtsp_stream_frames_generator(rtsp_url=rtsp_url):
        results = model(frame, verbose=False, device=device, conf=confidence_threshold)[
            0
        ]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == 0]

        annotated_frame = frame.copy()
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        for idx, polygon in enumerate(POLYGONS):
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame, polygon=polygon, color=COLORS.by_idx(idx)
            )

        cv2.imshow("Processed Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Streams video from an RTSP URL and performs object detection."
    )
    parser.add_argument(
        "--rtsp_url",
        type=str,
        required=True,
        help="Complete RTSP URL for the video stream.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8s.pt",
        help="Path to the model weights file. Default is 'yolov8s.pt'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device ('cpu', 'mps' or 'cuda'). Default is 'cpu'.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.3,
        help="Confidence level for detections (0 to 1). Default is 0.3.",
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.7,
        type=float,
        help="IOU threshold for non-max suppression. Default is 0.7.",
    )
    args = parser.parse_args()

    main(
        rtsp_url=args.rtsp_url,
        weights=args.weights,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )
