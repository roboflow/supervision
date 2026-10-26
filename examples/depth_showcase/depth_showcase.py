from __future__ import annotations

import contextlib
import itertools
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

import supervision as sv
from supervision.assets import VideoAssets, download_assets

STREAM_FPS = 30
CAPTION_COLOR = sv.Color.from_hex("#FFB450")
Predictor = Callable[[np.ndarray], sv.DepthMap]


def load_depth_anything(device: str) -> tuple[Predictor, str]:
    """Load Depth Anything V2 and return its frame predictor and caption."""
    from PIL import Image
    from transformers import pipeline

    estimator = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        device=device,
    )

    def predict(frame: np.ndarray) -> sv.DepthMap:
        """Predict relative inverse depth for one BGR frame."""
        result = estimator(Image.fromarray(frame[..., ::-1]))
        return sv.DepthMap.from_transformers(result, kind="relative_inverse")

    return predict, "Depth Anything V2 | relative depth"


def load_depth_pro(device: str) -> tuple[Predictor, str]:
    """Load Apple Depth Pro and return its frame predictor and caption."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, DepthProForDepthEstimation

    processor = AutoImageProcessor.from_pretrained("apple/DepthPro-hf")
    model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf")
    model = model.to(device).eval()

    def predict(frame: np.ndarray) -> sv.DepthMap:
        """Predict metric depth for one BGR frame."""
        image = Image.fromarray(frame[..., ::-1])
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)]
        )[0]
        return sv.DepthMap.from_transformers(result, kind="depth_m")

    return predict, "Apple Depth Pro | metric depth"


def load_yolo26(device: str) -> tuple[Predictor, str]:
    """Load YOLO26 depth and return its frame predictor and caption."""
    from ultralytics import YOLO

    model = YOLO("yolo26l-depth.pt")

    def predict(frame: np.ndarray) -> sv.DepthMap:
        """Predict metric depth for one BGR frame."""
        result = model(frame, verbose=False, device=device)[0]
        return sv.DepthMap.from_ultralytics(result)

    return predict, "YOLO26 Depth | metric depth"


MODELS = {
    "depth-anything": load_depth_anything,
    "depth-pro": load_depth_pro,
    "yolo26": load_yolo26,
}


def default_device() -> str:
    """Choose the fastest device that the installed PyTorch runtime can use."""
    try:
        import torch
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch.backends, "mps", None), "is_available", None)
    if callable(mps) and mps():
        return "mps"
    return "cpu"


@dataclass
class VideoSource:
    """A video file, or a live stream of unknown length such as a camera.

    Attributes:
        path: Video file path, camera index or stream URL
        fps: Frames per second of the video, assumed for a stream
        total_frames: Reported frame count, or `None` when it is unavailable
        max_frames: Number of frames to process, `None` for no limit
    """

    path: str
    fps: float
    total_frames: int | None
    max_frames: int | None

    @property
    def is_stream(self) -> bool:
        """Return whether the path names a camera or network stream."""
        return self.path.isdigit() or "://" in self.path

    def frames(self) -> Iterator[np.ndarray]:
        """Yield the frames of the file, or of the stream until it ends."""
        if not self.is_stream:
            yield from sv.get_video_frames_generator(self.path, end=self.max_frames)
            return
        import cv2

        capture = cv2.VideoCapture(int(self.path) if self.path.isdigit() else self.path)
        try:
            count = 0
            while self.max_frames is None or count < self.max_frames:
                success, frame = capture.read()
                if not success:
                    break
                count += 1
                yield frame
        finally:
            capture.release()

    def __post_init__(self) -> None:
        """Reject frame limits that cannot produce a useful video source."""
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be greater than zero.")


def sample_colour_range(
    predict: Predictor, source: VideoSource, samples: int
) -> sv.DepthClipRange | None:
    """Estimate a stable colour range from representative frames when possible.

    Known positive frame counts sample across the file. When metadata has no usable
    count, the first frames calibrate the range; unusable maps leave auto-ranging to
    :class:`~supervision.depth.annotators.DepthAnnotator`.
    """
    if source.is_stream:
        return None
    if samples <= 0:
        raise ValueError("range_samples must be greater than zero.")

    if source.total_frames is not None and source.total_frames > 0:
        frame_count = source.total_frames
        if source.max_frames is not None:
            frame_count = min(frame_count, source.max_frames)
        indexes = np.unique(
            np.linspace(0, frame_count - 1, min(samples, frame_count)).astype(int)
        )
        sample_frames = (
            next(
                sv.get_video_frames_generator(
                    source.path, start=int(index), end=int(index) + 1
                ),
                None,
            )
            for index in indexes
        )
    else:
        sample_count = min(samples, source.max_frames) if source.max_frames else samples
        sample_frames = itertools.islice(source.frames(), sample_count)

    depth_maps = [predict(frame) for frame in sample_frames if frame is not None]
    valid_maps = [depth_map for depth_map in depth_maps if depth_map.valid_mask.any()]
    if not valid_maps:
        return None
    return sv.DepthClipRange.from_depth_maps(valid_maps)


def blend_range(
    previous: tuple[float, float] | None, depth_map: sv.DepthMap, weight: float = 0.9
) -> tuple[float, float] | None:
    """Move the running colour range towards the range of this depth map.

    A live stream has no future frames to sample and a range per frame flickers, so the
    range follows the frames slowly instead.
    """
    if not depth_map.valid_mask.any():
        return previous
    low, high = sv.DepthClipRange.from_depth_maps([depth_map]).display_range
    if previous is None:
        return low, high
    return weight * previous[0] + (1 - weight) * low, weight * previous[1] + (
        1 - weight
    ) * high


def run_model(
    model_name: str,
    source: VideoSource,
    target_video_path: Path | None,
    colormap: str,
    opacity: float,
    range_samples: int,
    device: str,
) -> bool:
    """Predict, colour and write or show every frame in one pass.

    Returns False when the user closed the live window, to stop the remaining models.
    """
    frames = source.frames()
    first_frame = next(frames, None)
    if first_frame is None:
        raise ValueError(f"Video source {source.path!r} contains no readable frames.")
    predict, caption = MODELS[model_name](device)
    frames = itertools.chain([first_frame], frames)
    height, width = first_frame.shape[:2]
    fixed_range = None
    if not source.is_stream:
        fixed_range = sample_colour_range(predict, source, range_samples)
    elapsed_total = 0.0
    measured_frames = 0
    frame_index = 0
    running_range = None
    with contextlib.ExitStack() as stack:
        if target_video_path is not None:
            video_info = sv.VideoInfo(width, height, source.fps)
            sink = stack.enter_context(sv.VideoSink(str(target_video_path), video_info))
        else:
            window = sv.ImageWindow(model_name)
            stack.callback(window.close)
        frame_total = source.max_frames or source.total_frames
        for frame in tqdm(frames, total=frame_total, desc=f"Running {model_name}"):
            start = time.perf_counter()
            depth_map = predict(frame)
            duration = time.perf_counter() - start
            if frame_index > 0:
                elapsed_total += duration
                measured_frames += 1
            frame_index += 1
            if fixed_range is None:
                running_range = blend_range(running_range, depth_map)
            annotator = sv.DepthAnnotator(
                colormap=colormap,
                display_range=fixed_range or running_range or "auto",
                opacity=opacity,
            )
            frame = annotator.annotate(frame, depth_map)
            speed = (
                elapsed_total / measured_frames if measured_frames else duration
            ) * 1000
            frame = sv.draw_text(
                frame,
                f"{caption} | {speed:.0f} ms/frame",
                sv.Point(width // 2, height - 50),
                text_color=CAPTION_COLOR,
                text_scale=width / 1800,
                text_thickness=2,
                background_color=sv.Color.BLACK,
            )
            if target_video_path is not None:
                sink.write_frame(frame)
                continue
            window.show(frame)
            if not window.is_open or window.wait_key(1) == "q":
                return False
    return True


def main(
    source_video_path: str,
    target_directory: str | None = None,
    assets_directory: str = "data",
    models: list[str] = ["depth-anything", "depth-pro", "yolo26"],
    max_seconds: float | None = None,
    range_samples: int = 5,
    colormap: str = "turbo",
    opacity: float = 0.9,
    device: str = "cpu",
) -> None:
    """Colour a video or a live stream with the depth of every model.

    Every frame is predicted, coloured and written in one pass. Without a target
    directory the frames are shown in a window instead, and 'q' stops. For a video
    file the colour range is fixed from a few sampled frames, for a live stream it
    follows the frames.

    Args:
        source_video_path: Path to a video file, the name of a sample video of
            supervision such as 'VEHICLES' or 'PEOPLE_WALKING' which is downloaded
            first, a camera index such as '0', or a stream URL such as 'rtsp://...'
        target_directory: Directory for the output videos, written to
            `<target_directory>/<source name>-<model>.mp4`; the video is only shown
            live when not set
        assets_directory: Directory where sample videos are downloaded
        models: Models to run: 'depth-anything', 'depth-pro' or 'yolo26'
        max_seconds: Length of the video to process, the whole video or stream if
            not set
        range_samples: Number of video frames sampled to fix its colour range
        colormap: Colour table of the depth maps, such as 'turbo' or 'viridis'
        opacity: Opacity of the depth colours over the video (0 to 1)
        device: Computation device ('cpu', 'mps' or 'cuda')
    """
    if range_samples <= 0:
        raise ValueError("range_samples must be greater than zero.")
    is_stream = source_video_path.isdigit() or "://" in source_video_path
    if not is_stream and source_video_path.upper() in VideoAssets.__members__:
        asset = VideoAssets[source_video_path.upper()]
        source_video_path = download_assets(asset, assets_directory)
    fps, total_frames = STREAM_FPS, None
    if not is_stream:
        video_info = sv.VideoInfo.from_video_path(source_video_path)
        fps = video_info.fps
        total_frames = video_info.total_frames
        if total_frames is not None and total_frames <= 0:
            total_frames = None
    max_frames = None
    if max_seconds is not None:
        if not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("max_seconds must be a finite number greater than zero.")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(
                "Video FPS must be a positive finite number when max_seconds is set."
            )
        max_frames = int(max_seconds * fps)
        if max_frames <= 0:
            raise ValueError(
                "max_seconds must include at least one frame at the source FPS."
            )
        if total_frames is not None and total_frames > 0:
            max_frames = min(max_frames, total_frames)
    source = VideoSource(source_video_path, fps, total_frames, max_frames)
    if target_directory is not None:
        Path(target_directory).mkdir(parents=True, exist_ok=True)

    for model_name in models:
        target_video_path = None
        if target_directory is not None:
            stem = "stream" if is_stream else Path(source_video_path).stem
            target_video_path = Path(target_directory) / f"{stem}-{model_name}.mp4"
        completed = run_model(
            model_name,
            source,
            target_video_path,
            colormap,
            opacity,
            range_samples,
            device,
        )
        if not completed:
            break


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
