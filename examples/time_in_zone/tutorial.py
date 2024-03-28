import argparse

import cv2

import numpy as np
import supervision as sv
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from utils.general import load_zones_config, get_stream_frames_generator
from utils.timers import ClockBasedTimer


COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(color=COLORS)


class CustomSink:
    def __init__(self, zone_configuration_path: str, video_sink: sv.VideoSink):
        self.video_sink = video_sink
        self.tracker = sv.ByteTrack()
        self.polygons = load_zones_config(file_path=zone_configuration_path)
        self.timers = [ClockBasedTimer() for _ in self.polygons]
        self.zones = None

    def on_prediction(self, result: dict, frame: VideoFrame) -> None:
        if self.zones is None:
            resolution_wh = frame.image.shape[1], frame.image.shape[0]
            self.zones = [
                sv.PolygonZone(
                    polygon=polygon,
                    frame_resolution_wh=resolution_wh,
                    triggering_anchors=(sv.Position.CENTER,),
                )
                for polygon in self.polygons
            ]

        detections = sv.Detections.from_inference(result)
        detections = detections[detections.class_id == 0]
        detections = self.tracker.update_with_detections(detections=detections)

        annotated_frame = frame.image.copy()
        for idx, zone in enumerate(self.zones):
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame,
                polygon=zone.polygon,
                color=COLORS.by_idx(idx),
                thickness=2
            )
            detections_in_zone = detections[zone.trigger(detections)]
            time_in_zone = self.timers[idx].tick(detections_in_zone)
            custom_color_lookup = np.full_like(detections_in_zone.class_id, idx)
            annotated_frame = BOUNDING_BOX_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                custom_color_lookup=custom_color_lookup
            )
            labels = [
                f"#{tracker_id} {time:.1f}s"
                for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
            ]
            annotated_frame = LABEL_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                labels=labels,
                custom_color_lookup=custom_color_lookup
            )

        self.video_sink.write_frame(annotated_frame)
        cv2.imshow("Result", annotated_frame)
        cv2.waitKey(1)


def main(
    rtsp_url: str,
    zone_configuration_path: str,
    model_id: str
) -> None:
    video_info = sv.VideoInfo(
        fps=30,
        width=1280,
        height=720
    )
    with sv.VideoSink('data/output.mp4', video_info=video_info) as video_sink:
        sink = CustomSink(zone_configuration_path=zone_configuration_path, video_sink=video_sink)

        pipeline = InferencePipeline.init(
            model_id=model_id,
            video_reference=rtsp_url,
            on_prediction=sink.on_prediction
        )

        pipeline.start()

        try:
            pipeline.join()
        except KeyboardInterrupt:
            pipeline.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculating detections dwell time in zones, using video file."
    )
    parser.add_argument(
        "--zone_configuration_path",
        type=str,
        required=True,
        help="Path to the zone configuration JSON file.",
    )
    parser.add_argument(
        "--rtsp_url",
        type=str,
        required=True,
        help="Complete RTSP URL for the video stream.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="yolov8s-640",
        help="Roboflow model ID."
    )
    args = parser.parse_args()

    main(
        rtsp_url=args.rtsp_url,
        zone_configuration_path=args.zone_configuration_path,
        model_id=args.model_id
    )
