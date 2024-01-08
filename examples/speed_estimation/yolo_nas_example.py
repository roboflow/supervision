import argparse

from super_gradients.training import models
from super_gradients.common.object_names import Models

import supervision as sv


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Supervision Package"
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        required=True,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.7,
        help="IOU threshold for the model",
        type=float
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)
    model = models.get(Models.YOLO_NAS_L, pretrained_weights="coco")