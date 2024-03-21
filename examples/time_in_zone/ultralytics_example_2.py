from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

import supervision as sv

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])

POLYGONS = [
    np.array(
        [
            [685, 113],
            [845, 110],
            [888, 188],
            [879, 294],
            [912, 415],
            [925, 344],
            [981, 511],
            [944, 620],
            [728, 640],
        ]
    ),
    np.array(
        [[964, 101], [1124, 332], [1205, 480], [1146, 588], [1280, 558], [1280, 97]]
    ),
    np.array(
        [
            [402, 137],
            [499, 138],
            [471, 209],
            [494, 312],
            [479, 419],
            [465, 378],
            [434, 540],
            [478, 648],
            [299, 641],
            [226, 539],
            [279, 387],
        ]
    ),
]

PIXELATE_ANNOTATOR = sv.PixelateAnnotator(pixel_size=10)
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)


class Timer:
    def __init__(self, fps: int = 30) -> None:
        self.fps = fps
        self.frame_id = 0
        self.tacker_id2frame_id: Dict[int, int] = {}

    def tick(self, detections: sv.Detections) -> None:
        self.frame_id += 1
        for tracker_id in detections.tracker_id:
            self.tacker_id2frame_id.setdefault(tracker_id, self.frame_id)

    def get_time(self, tracker_id: int) -> Optional[float]:
        start_frame_id = self.tacker_id2frame_id.get(tracker_id, None)
        if start_frame_id is None:
            return None
        return (self.frame_id - start_frame_id) / self.fps


def initiate_polygon_zones(
    polygons: List[np.ndarray],
    frame_resolution_wh: Tuple[int, int],
    triggering_anchors: Iterable[sv.Position] = [sv.Position.CENTER],
) -> List[sv.PolygonZone]:
    return [
        sv.PolygonZone(
            polygon=polygon,
            frame_resolution_wh=frame_resolution_wh,
            triggering_anchors=triggering_anchors,
        )
        for polygon in polygons
    ]


def main(source_video_path: str, weights: str, device: str, confidence: float) -> None:
    tracker = sv.ByteTrack()
    smoother = sv.DetectionsSmoother(length=10)
    model = YOLO(weights)
    face_model = YOLO("data/face.pt")

    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    frame = next(frame_generator)
    frame_resolution_wh = frame.shape[1], frame.shape[0]
    zones = initiate_polygon_zones(
        polygons=POLYGONS, frame_resolution_wh=frame_resolution_wh
    )
    timers = [Timer(fps=video_info.fps) for _ in zones]

    with sv.VideoSink("data/target/checkout-new-zones-2.mp4", video_info) as sink:
        for frame in frame_generator:
            face_results = face_model(
                frame, verbose=False, device=device, conf=confidence
            )[0]
            face_detections = sv.Detections.from_ultralytics(face_results)

            results = model(frame, verbose=False, device=device, conf=confidence)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[detections.class_id == 0]
            detections = detections.with_nms(threshold=0.2)
            detections = tracker.update_with_detections(detections=detections)
            detections = smoother.update_with_detections(detections=detections)

            annotated_frame = frame.copy()
            annotated_frame = PIXELATE_ANNOTATOR.annotate(
                scene=annotated_frame, detections=face_detections
            )

            for idx, zone in enumerate(zones):
                annotated_frame = sv.draw_polygon(
                    scene=annotated_frame,
                    polygon=zone.polygon,
                    color=COLORS.by_idx(idx),
                )
                detections_in_zone = detections[zone.trigger(detections)]
                detections_in_zone.class_id = np.full(
                    detections_in_zone.class_id.shape, idx
                )
                timers[idx].tick(detections_in_zone)

                annotated_frame = COLOR_ANNOTATOR.annotate(
                    scene=annotated_frame, detections=detections_in_zone
                )
                labels = [
                    f"#{tracker_id} {int(timers[idx].get_time(tracker_id) // 60):02d}:{int(timers[idx].get_time(tracker_id) % 60):02d}"
                    for tracker_id in detections_in_zone.tracker_id
                ]
                annotated_frame = LABEL_ANNOTATOR.annotate(
                    scene=annotated_frame, detections=detections_in_zone, labels=labels
                )

            sink.write_frame(annotated_frame)
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
        "--source_video_path",
        type=str,
        required=True,
        help="Path to the source video file",
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
        "--confidence",
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
        source_video_path=args.source_video_path,
        weights=args.weights,
        device=args.device,
        confidence=args.confidence,
    )
