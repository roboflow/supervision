from __future__ import annotations

import contextlib
import itertools
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import supervision as sv
from supervision.assets import VideoAssets, download_assets

STREAM_FPS = 30
CAPTION_COLOR = sv.Color.from_hex("#FFB450")
Predictor = Callable[[np.ndarray], sv.DepthMap]


def load_depth_anything(device: str) -> tuple[Predictor, str]:
    from PIL import Image
    from transformers import pipeline

    estimator = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        device=device,
    )

    def predict(frame: np.ndarray) -> sv.DepthMap:
        result = estimator(Image.fromarray(frame[..., ::-1]))
        return sv.DepthMap.from_transformers(result, kind="relative_inverse")

    return predict, "Depth Anything V2 | relative depth"


def load_depth_pro(device: str) -> tuple[Predictor, str]:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, DepthProForDepthEstimation

    processor = AutoImageProcessor.from_pretrained("apple/DepthPro-hf")
    model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf")
    model = model.to(device).eval()

    def predict(frame: np.ndarray) -> sv.DepthMap:
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
    from ultralytics import YOLO

    model = YOLO("yolo26l-depth.pt")

    def predict(frame: np.ndarray) -> sv.DepthMap:
        result = model(frame, verbose=False, device=device)[0]
        return sv.DepthMap.from_ultralytics(result)

    return predict, "YOLO26 Depth | metric depth"


MODELS = {
    "depth-anything": load_depth_anything,
    "depth-pro": load_depth_pro,
    "yolo26": load_yolo26,
}


@dataclass
class VideoSource:
    """A video file, or a live stream of unknown length such as a camera.

    Attributes:
        path: Video file path, camera index or stream URL
        fps: Frames per second of the video, assumed for a stream
        total_frames: Frame count of a video file, `None` for a live stream
        max_frames: Number of frames to process, `None` for no limit
    """

    path: str
    fps: float
    total_frames: int | None
    max_frames: int | None

    @property
    def is_stream(self) -> bool:
        return self.total_frames is None

    def frames(self) -> Iterator[np.ndarray]:
        """Yield the frames of the file, or of the stream until it ends."""
        if not self.is_stream:
            yield from sv.get_video_frames_generator(self.path, end=self.max_frames)
            return
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


def sample_colour_range(
    predict: Predictor, source: VideoSource, samples: int
) -> sv.DepthClipRange:
    """Predict a few frames spread from the first to the last and take their range.

    A video file is known up front, so the colour range of the whole clip is estimated
    from the first frame, the last frame and the ones in between.
    """
    if source.total_frames is None:
        raise ValueError("A live stream has no frames to sample ahead of time.")
    last = min(source.total_frames, source.max_frames or source.total_frames) - 1
    depth_maps = []
    for index in np.linspace(0, last, samples).astype(int):
        frames = sv.get_video_frames_generator(
            source.path, start=int(index), end=int(index) + 1
        )
        depth_maps.append(predict(next(frames)))
    return sv.DepthClipRange.from_depth_maps(depth_maps)


def blend_range(
    previous: tuple[float, float] | None, depth_map: sv.DepthMap, weight: float = 0.9
) -> tuple[float, float]:
    """Move the running colour range towards the range of this depth map.

    A live stream has no future frames to sample and a range per frame flickers, so the
    range follows the frames slowly instead.
    """
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
    predict, caption = MODELS[model_name](device)
    frames = source.frames()
    first_frame = next(frames)
    frames = itertools.chain([first_frame], frames)
    height, width = first_frame.shape[:2]
    fixed_range = None
    if not source.is_stream:
        fixed_range = sample_colour_range(predict, source, range_samples)
    durations: list[float] = []
    running_range = None
    with contextlib.ExitStack() as stack:
        if target_video_path is not None:
            video_info = sv.VideoInfo(width, height, source.fps)
            sink = stack.enter_context(sv.VideoSink(str(target_video_path), video_info))
        else:
            window = sv.ImageWindow(model_name)
            stack.callback(window.close)
        for frame in tqdm(
            frames, total=source.max_frames, desc=f"Running {model_name}"
        ):
            start = time.perf_counter()
            depth_map = predict(frame)
            durations.append(time.perf_counter() - start)
            running_range = blend_range(running_range, depth_map)
            annotator = sv.DepthAnnotator(
                colormap=colormap,
                display_range=fixed_range or running_range,
                opacity=opacity,
            )
            frame = annotator.annotate(frame, depth_map)
            # the first call pays one-off warm-up costs
            speed = np.mean(durations[1:] or durations) * 1000
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
        range_samples: Frames sampled from the first to the last of a video file to
            fix its colour range
        colormap: Colour table of the depth maps, such as 'turbo' or 'viridis'
        opacity: Opacity of the depth colours over the video (0 to 1)
        device: Computation device ('cpu', 'mps' or 'cuda')
    """
    is_stream = source_video_path.isdigit() or "://" in source_video_path
    if not is_stream and source_video_path.upper() in VideoAssets.__members__:
        asset = VideoAssets[source_video_path.upper()]
        source_video_path = download_assets(asset, assets_directory)
    fps, total_frames = STREAM_FPS, None
    if not is_stream:
        video_info = sv.VideoInfo.from_video_path(source_video_path)
        fps, total_frames = video_info.fps, video_info.total_frames
    max_frames = total_frames
    if max_seconds is not None:
        max_frames = int(max_seconds * fps)
        if total_frames is not None:
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
