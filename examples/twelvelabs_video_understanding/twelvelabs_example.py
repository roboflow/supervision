from __future__ import annotations

import os

import cv2
import numpy as np

import supervision as sv

COLORS = sv.ColorPalette.DEFAULT
BOX_ANNOTATOR = sv.BoxAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(color=COLORS)


def analyze_video(
    source_video_path: str,
    prompt: str,
    model_name: str = "pegasus1.5",
    max_tokens: int = 512,
    api_key: str = "",
) -> str:
    """
    Generate a natural-language understanding of a whole video with TwelveLabs
    Pegasus.

    Args:
        source_video_path: Path to the source video file to analyze.
        prompt: Instruction passed to Pegasus, e.g. "Summarize this video".
        model_name: TwelveLabs Pegasus model to use.
        max_tokens: Maximum number of tokens in the generated response.
        api_key: TwelveLabs API key. Falls back to the `TWELVELABS_API_KEY`
            environment variable when empty.

    Returns:
        The text produced by Pegasus describing the video.
    """
    import base64

    from twelvelabs import TwelveLabs
    from twelvelabs.types.video_context import VideoContext_Base64String

    api_key = api_key or os.environ.get("TWELVELABS_API_KEY", "")
    if not api_key:
        raise ValueError(
            "A TwelveLabs API key is required. Pass --api_key or set the "
            "TWELVELABS_API_KEY environment variable. Grab a free key at "
            "https://twelvelabs.io."
        )

    with open(source_video_path, "rb") as video_file:
        encoded_video = base64.b64encode(video_file.read()).decode("utf-8")

    client = TwelveLabs(api_key=api_key)
    response = client.analyze(
        model_name=model_name,
        video=VideoContext_Base64String(base_64_string=encoded_video),
        prompt=prompt,
        max_tokens=max_tokens,
    )
    return response.data or ""


def main(
    source_video_path: str,
    prompt: str = "Summarize what happens in this video.",
    model_id: str = "yolov8n.pt",
    confidence_threshold: float = 0.3,
    pegasus_model_name: str = "pegasus1.5",
    max_tokens: int = 512,
    target_video_path: str | None = None,
    api_key: str = "",
) -> None:
    """
    Combine per-frame Supervision detections with a whole-video TwelveLabs
    Pegasus understanding.

    Supervision answers "what object is where, in each frame" while Pegasus
    answers "what is this video about". Running both on the same clip pairs
    precise, frame-level detections with a high-level narrative summary.

    Args:
        source_video_path: Path to the source video file.
        prompt: Instruction passed to Pegasus for the video-level summary.
        model_id: Ultralytics YOLO model id or weights path for detection.
        confidence_threshold: Confidence level for detections (0 to 1).
        pegasus_model_name: TwelveLabs Pegasus model to use.
        max_tokens: Maximum number of tokens in the Pegasus response.
        target_video_path: Optional path to save the annotated video. When
            omitted, annotated frames are displayed in a window.
        api_key: TwelveLabs API key. Falls back to the `TWELVELABS_API_KEY`
            environment variable when empty.
    """
    from ultralytics import YOLO

    summary = analyze_video(
        source_video_path=source_video_path,
        prompt=prompt,
        model_name=pegasus_model_name,
        max_tokens=max_tokens,
        api_key=api_key,
    )
    print("TwelveLabs Pegasus summary:")
    print(summary)

    model = YOLO(model_id)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    frames_generator = sv.get_video_frames_generator(source_video_path)

    def annotate(frame: np.ndarray) -> np.ndarray:
        result = model(frame, conf=confidence_threshold, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        annotated_frame = BOX_ANNOTATOR.annotate(frame.copy(), detections)
        annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        return annotated_frame

    if target_video_path is not None:
        with sv.VideoSink(target_video_path, video_info) as sink:
            for frame in frames_generator:
                sink.write_frame(annotate(frame))
    else:
        for frame in frames_generator:
            cv2.imshow("Processed Video", annotate(frame))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
