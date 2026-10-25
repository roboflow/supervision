from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from tqdm import tqdm

import supervision as sv
from supervision.assets import VideoAssets, download_assets

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


def estimate_depth(
    model_name: str, source_video_path: str, max_frames: int, device: str
) -> tuple[list[sv.DepthMap], str]:
    """Run one model on the first `max_frames` frames of the video.

    Returns the depth maps, shrunk to keep a long clip in memory, and the caption of the
    model with its measured speed.
    """
    predict, caption = MODELS[model_name](device)
    frames = sv.get_video_frames_generator(source_video_path, end=max_frames)
    depth_maps, durations = [], []
    for frame in tqdm(frames, total=max_frames, desc=f"Running {model_name}"):
        start = time.perf_counter()
        depth_map = predict(frame)
        durations.append(time.perf_counter() - start)
        height, width = frame.shape[:2]
        depth_maps.append(depth_map.resize((640, 640 * height // width)))
    # the first call pays one-off warm-up costs
    speed = np.mean(durations[1:] or durations) * 1000
    return depth_maps, f"{caption} | {speed:.0f} ms/frame"


def write_depth_video(
    source_video_path: str,
    target_video_path: str,
    depth_maps: list[sv.DepthMap],
    caption: str,
    colormap: str,
    opacity: float,
) -> None:
    """Write the source video coloured with the depth maps and captioned."""
    annotator = sv.DepthAnnotator(
        colormap=colormap,
        # one colour range for the whole clip keeps the colours of a still wall steady
        display_range=sv.DepthClipRange.from_depth_maps(depth_maps),
        opacity=opacity,
    )

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        frame = annotator.annotate(frame, depth_maps[index])
        height, width = frame.shape[:2]
        return sv.draw_text(
            frame,
            caption,
            sv.Point(width // 2, height - 50),
            text_color=CAPTION_COLOR,
            text_scale=width / 1800,
            text_thickness=2,
            background_color=sv.Color.BLACK,
        )

    sv.process_video(
        source_path=source_video_path,
        target_path=target_video_path,
        callback=callback,
        max_frames=len(depth_maps),
        show_progress=True,
        progress_message=f"Writing {Path(target_video_path).name}",
    )


def main(
    source_video_path: str,
    target_directory: str = "outputs",
    assets_directory: str = "data",
    models: list[str] = ["depth-anything", "depth-pro", "yolo26"],
    max_seconds: float | None = None,
    colormap: str = "turbo",
    opacity: float = 0.9,
    device: str = "cpu",
) -> None:
    """Colour a video with the depth of every model, one output video per model.

    Each video is written as soon as its model has finished, to
    `<target_directory>/<source name>-<model>.mp4`.

    Args:
        source_video_path: Path to the source video file, or the name of a sample
            video of supervision such as 'VEHICLES' or 'PEOPLE_WALKING', which is
            downloaded first
        target_directory: Directory for the output videos
        assets_directory: Directory where sample videos are downloaded
        models: Models to run: 'depth-anything', 'depth-pro' or 'yolo26'
        max_seconds: Length of the video to process, the whole video if not set
        colormap: Colour table of the depth maps, such as 'turbo' or 'viridis'
        opacity: Opacity of the depth colours over the video (0 to 1)
        device: Computation device ('cpu', 'mps' or 'cuda')
    """
    if source_video_path.upper() in VideoAssets.__members__:
        asset = VideoAssets[source_video_path.upper()]
        source_video_path = download_assets(asset, assets_directory)
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    max_frames = video_info.total_frames
    if max_seconds is not None:
        max_frames = min(max_frames, int(max_seconds * video_info.fps))
    Path(target_directory).mkdir(parents=True, exist_ok=True)

    for model_name in models:
        depth_maps, caption = estimate_depth(
            model_name, source_video_path, max_frames, device
        )
        target_video_path = Path(target_directory) / (
            f"{Path(source_video_path).stem}-{model_name}.mp4"
        )
        write_depth_video(
            source_video_path,
            str(target_video_path),
            depth_maps,
            caption,
            colormap,
            opacity,
        )


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
