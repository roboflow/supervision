import os
from contextlib import ExitStack, suppress

import numpy as np
from trackers import ByteTrackTracker
from ultralytics import YOLO
from utils.general import find_in_list, load_zones_config
from utils.timers import FPSBasedTimer

import supervision as sv

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)
#: Output video codecs tried in order; the first one OpenCV can open is used.
VIDEO_CODECS = ("H264", "XVID")


def _is_same_file(output_path: str, source_path: str) -> bool:
    """Tell whether `output_path` already exists as the same file as `source_path`.

    Opening a video writer truncates an existing target, so an output that is the
    source file, through a symlink, a hard link or a differently spelled path,
    would destroy the video before it is read. A missing output cannot be the
    source, which must exist.

    Args:
        output_path: Path the annotated video would be written to.
        source_path: Path of the video being read.

    Returns:
        `True` if both paths refer to the same existing file.
    """
    return os.path.exists(output_path) and os.path.samefile(output_path, source_path)


def _open_video_sink(
    stack: ExitStack, output_video_path: str, video_info: sv.VideoInfo
) -> sv.VideoSink:
    """Open a video sink with the first codec in `VIDEO_CODECS` that works.

    OpenCV does not raise when a codec or output path is unusable, so a failed
    candidate surfaces as a `RuntimeError` from `sv.VideoSink`, which releases the
    unusable writer itself. The opened sink is registered on `stack`, so it is
    released when the stack exits.

    Args:
        stack: Exit stack that owns the opened sink.
        output_video_path: Path of the video file to write.
        video_info: Resolution and FPS of the video to write.

    Returns:
        The opened video sink.

    Raises:
        RuntimeError: If none of `VIDEO_CODECS` can be opened for the path.
    """
    for codec in VIDEO_CODECS:
        with suppress(RuntimeError):
            video_sink = stack.enter_context(
                sv.VideoSink(output_video_path, video_info, codec=codec)
            )
            print(f"Video output being saved: {output_video_path}")
            # The container may remap the requested codec (e.g. XVID to mp4v for
            # .mp4), so report what was requested, not what is written.
            print(f"Requested codec: {codec} (the container may remap it)")
            return video_sink
    raise RuntimeError(
        f"Could not open a video writer for {output_video_path!r} with any "
        f"of the codecs {VIDEO_CODECS}. Check that the output directory "
        "exists and that OpenCV was built with an encoder for one of them."
    )


def main(
    zone_configuration_path: str,
    source_video_path: str,
    weights: str = "yolov8s.pt",
    device: str = "cpu",
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
    classes: list[int] = [],
    output_video_path: str | None = None,
) -> None:
    """Calculating detections dwell time in zones, using video file.

    Args:
        zone_configuration_path: Path to the zone configuration JSON file
        source_video_path: Path to the source video file
        weights: Path to the model weights file
        device: Computation device ('cpu', 'mps' or 'cuda')
        confidence_threshold: Confidence level for detections (0 to 1)
        iou_threshold: IOU threshold for non-max suppression
        classes: List of class IDs to track. If empty, all classes are tracked
        output_video_path: Path to save the output video. If not provided, video
            will only be displayed.

    Raises:
        ValueError: If `output_video_path` is the same file as
            `source_video_path`.
        RuntimeError: If `output_video_path` is given but no video writer can be
            opened for it, for example because its directory does not exist.
    """
    if output_video_path and _is_same_file(output_video_path, source_video_path):
        raise ValueError(
            f"output_video_path {output_video_path!r} is the same file as "
            f"source_video_path {source_video_path!r}; writing would overwrite the "
            "source video. Choose a different output path."
        )
    model = YOLO(weights)
    tracker = ByteTrackTracker(
        minimum_iou_threshold=0.5, track_activation_threshold=confidence_threshold
    )
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    frames_generator = sv.get_video_frames_generator(source_video_path)

    polygons = load_zones_config(file_path=zone_configuration_path)
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=(sv.Position.CENTER,),
        )
        for polygon in polygons
    ]
    timers = [FPSBasedTimer(video_info.fps) for _ in zones]

    video_sink: sv.VideoSink | None = None
    # The stack releases the writer and closes the window on normal end, on `q`, and
    # when inference or the user interrupts the loop with an exception.
    with ExitStack() as stack:
        if output_video_path:
            video_sink = _open_video_sink(stack, output_video_path, video_info)
        window = stack.enter_context(sv.ImageWindow("Processed Video"))
        for frame in frames_generator:
            results = model(
                frame,
                verbose=False,
                device=device,
                conf=confidence_threshold,
                iou=iou_threshold,
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[find_in_list(detections.class_id, classes)]
            detections = tracker.update(detections)
            detections = detections[detections.tracker_id != -1]  # -1 = pending track

            annotated_frame = frame.copy()

            for idx, zone in enumerate(zones):
                annotated_frame = sv.draw_polygon(
                    scene=annotated_frame,
                    polygon=zone.polygon,
                    color=COLORS.by_idx(idx),
                )

                detections_in_zone = detections[zone.trigger(detections)]
                time_in_zone = timers[idx].tick(detections_in_zone)
                custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

                annotated_frame = COLOR_ANNOTATOR.annotate(
                    scene=annotated_frame,
                    detections=detections_in_zone,
                    custom_color_lookup=custom_color_lookup,
                )
                labels = [
                    f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                    for tracker_id, time in zip(
                        detections_in_zone.tracker_id, time_in_zone
                    )
                ]
                annotated_frame = LABEL_ANNOTATOR.annotate(
                    scene=annotated_frame,
                    detections=detections_in_zone,
                    labels=labels,
                    custom_color_lookup=custom_color_lookup,
                )

            window.show(annotated_frame)

            # Save frame to output video if a sink is available
            if video_sink is not None:
                video_sink.write_frame(annotated_frame)

            key = window.wait_key(1)
            if not window.is_open or key == "q":
                break

    # Reached only when the loop ended normally or on `q`, after the sink is released
    if video_sink is not None:
        print(f"Video successfully saved: {output_video_path}")


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
