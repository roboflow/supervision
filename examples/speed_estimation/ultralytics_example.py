import cv2
import argparse
import supervision as sv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Supervision Package"
    )

    parser.add_argument(
        "--lines_configuration_path",
        required=True,
        help="Path to the lines configuration JSON file",
        type=str,
    )
    parser.add_argument(
        "--source_weights_path",
        required=True,
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        default=None,
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

    args = parser.parse_args()

    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)
    frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)

    for frame in frame_generator:
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
