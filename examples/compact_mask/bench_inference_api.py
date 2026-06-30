"""Benchmark dense vs compact Roboflow RLE ingestion.

Run with:
    uv run python examples/compact_mask/bench_inference_api.py

The benchmark downloads supervision assets, runs one segmentation inference per
source image, then times dense vs compact parsing of that fixed inference result.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table

import supervision as sv
from supervision.assets import ImageAssets, VideoAssets, download_assets
from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.detection.compact_mask import CompactMask

console = Console(width=120, force_terminal=True)

# Default segmentation model; use an rfdetr-seg-* id so masks are returned.
MODEL_ID = "rfdetr-seg-large"
# Environment variable that can override MODEL_ID without adding CLI noise.
MODEL_ID_ENV = "BENCH_INFERENCE_MODEL_ID"
# Optional Roboflow API key for models that require authentication.
API_KEY_ENV = "ROBOFLOW_API_KEY"
# Model confidence threshold used only for the one inference call per source.
CONFIDENCE = 0.2
# Model IoU threshold used only for the one inference call per source.
IOU = 0.5
# Request native RLE masks so the benchmark measures RLE parser ingestion.
RESPONSE_MASK_FORMAT = "rle"
# Parser timing repetitions; inference itself is not repeated.
REPETITIONS = 50
# Untimed parser warmup calls before measurements.
WARMUP = 3
# Visual segmentation overlays for manual validation.
ARTIFACT_DIR = Path("examples/compact_mask/outputs")

ASSETS = {Path(asset.filename).stem: asset for asset in ImageAssets}
for video_asset in VideoAssets:
    key = Path(video_asset.filename).stem
    ASSETS[key if key not in ASSETS else f"{key}-video"] = video_asset


@dataclass
class ApiBenchmarkResult:
    """Result for one dense-vs-compact parser benchmark run."""

    source: str
    resolution: str
    segmented_objects: int
    dense_s: float
    compact_s: float
    dense_peak_bytes: int
    compact_peak_bytes: int
    dense_mask_bytes: int
    compact_mask_bytes: int
    pixel_perfect: bool


def load_image_from_asset(path: Path | None, asset: str) -> tuple[np.ndarray, str]:
    """Return ``(image, label)`` for an image or video middle frame."""
    if path is not None:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return image, str(path)

    asset_obj = ASSETS[asset]
    asset_path = Path(download_assets(asset_obj))
    if isinstance(asset_obj, ImageAssets):
        image = cv2.imread(str(asset_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {asset_path}")
        return image, str(asset_path)

    video = cv2.VideoCapture(str(asset_path))
    if not video.isOpened():
        raise FileNotFoundError(f"Could not read video: {asset_path}")
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = max(0, frame_count // 2)
    if frame_index:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = video.read()
    video.release()
    if not ok or frame is None:
        raise FileNotFoundError(f"Could not read middle frame: {asset_path}")
    return frame, f"{asset_path}#{frame_index}"


def freeze_result(inference_result: Any) -> dict[str, Any]:
    """Convert one Inference result to a reusable dictionary."""
    if isinstance(inference_result, dict):
        return inference_result
    if hasattr(inference_result, "model_dump"):
        return inference_result.model_dump(exclude_none=True, by_alias=True)
    if hasattr(inference_result, "dict"):
        return inference_result.dict(exclude_none=True, by_alias=True)
    raise TypeError(
        f"Expected dict-like Inference result, got {type(inference_result).__name__}"
    )


def count_rle_predictions(result: dict[str, Any]) -> int:
    """Return the number of predictions carrying Roboflow RLE masks."""
    return sum(
        isinstance(prediction.get("rle") or prediction.get("rle_mask"), dict)
        for prediction in result.get("predictions", [])
    )


def synthetic_dense_small_result() -> tuple[np.ndarray, str, dict[str, Any]]:
    """Return a small dense-mask adversarial payload where compact parsing is slower.

    Uses a 64x64 image with 4 fully-filled masks. At this scale the dense
    ``(N, H, W)`` allocation cost is negligible; Python RLE arithmetic dominates,
    making compact ingestion slower than the dense NumPy path. Included as a
    clearly labeled adversarial row in the default benchmark run to show that
    the ``speedup`` column reflects allocation savings, not decode speed.
    """
    height, width = 64, 64
    image = np.zeros((height, width, 3), dtype=np.uint8)
    predictions = [
        {
            "x": width / 2,
            "y": height / 2,
            "width": width,
            "height": height,
            "confidence": 0.9,
            "class_id": index,
            "class": f"dense-{index}",
            "rle": {"size": [height, width], "counts": [0, height * width]},
        }
        for index in range(4)
    ]
    return (
        image,
        "synthetic-dense-64",
        {
            "predictions": predictions,
            "image": {"width": width, "height": height},
        },
    )


def derive_boxes_from_rle_masks(result: dict[str, Any]) -> dict[str, Any]:
    """Set prediction boxes from native RLE segmentation masks."""
    predictions = []
    for prediction in result.get("predictions", []):
        rle = prediction.get("rle") or prediction.get("rle_mask")
        if not isinstance(rle, dict):
            predictions.append(prediction)
            continue

        height, width = rle["size"]
        mask = sv.rle_to_mask(rle["counts"], resolution_wh=(int(width), int(height)))
        if not mask.any():
            predictions.append(prediction)
            continue

        x1, y1, x2, y2 = sv.mask_to_xyxy(mask[np.newaxis, ...])[0]
        predictions.append(
            {
                **prediction,
                "x": float((x1 + x2) / 2),
                "y": float((y1 + y2) / 2),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
            }
        )
    return {**result, "predictions": predictions}


def artifact_path(source: str) -> Path:
    """Return the segmentation validation artifact path for a source."""
    source_path, separator, frame = source.partition("#")
    stem = Path(source_path).stem
    suffix = f"_frame_{frame}" if separator else ""
    return ARTIFACT_DIR / f"{stem}{suffix}_segmentations.jpg"


def detection_labels(detections: sv.Detections) -> list[str]:
    """Return compact class/confidence labels for validation artifacts."""
    raw_class_names = detections.get_data(CLASS_NAME_DATA_FIELD)
    class_names = (
        raw_class_names.astype(str).tolist()
        if isinstance(raw_class_names, np.ndarray)
        else [""] * len(detections)
    )

    labels = []
    for index in range(len(detections)):
        class_name = class_names[index] if index < len(class_names) else ""
        confidence = (
            ""
            if detections.confidence is None
            else f" {detections.confidence[index]:.2f}"
        )
        labels.append(f"{class_name}{confidence}".strip() or str(index))
    return labels


def save_segmentation_artifact(
    image: np.ndarray,
    result: dict[str, Any],
    source: str,
) -> Path | None:
    """Draw parsed segmentation masks and save a validation artifact."""
    detections = sv.Detections.from_inference(result)
    if detections.mask is None:
        return None

    annotated = image.copy()
    annotated = sv.MaskAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        opacity=0.45,
    ).annotate(scene=annotated, detections=detections)
    annotated = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        text_scale=0.35,
        text_padding=4,
    ).annotate(
        scene=annotated,
        detections=detections,
        labels=detection_labels(detections),
    )

    path = artifact_path(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), annotated):
        raise OSError(f"Could not write segmentation artifact: {path}")
    return path


def load_inference_model(model_id: str, api_key: str | None) -> Any:
    """Load the requested Inference model."""
    try:
        from inference import get_model
    except ImportError as exc:
        raise ImportError(
            "Install the `inference` package to run this benchmark."
        ) from exc

    model_kwargs = {"api_key": api_key} if api_key is not None else {}
    return get_model(model_id=model_id, **model_kwargs)


def run_inference_once(
    image: np.ndarray,
    model: Any,
    model_id: str,
    confidence: float,
    iou: float,
) -> dict[str, Any] | None:
    """Run one real segmentation inference and return a frozen result."""
    # Inference still serializes instance segmentations with x/y/width/height.
    # Derive those fields from the RLE masks so the benchmark uses segmentations,
    # not the model-reported detector boxes, as the source of truth.
    result = derive_boxes_from_rle_masks(
        freeze_result(
            model.infer(
                image,
                confidence=confidence,
                iou=iou,
                response_mask_format=RESPONSE_MASK_FORMAT,
            )[0]
        )
    )
    rle_count = count_rle_predictions(result)
    if rle_count == 0:
        console.print(
            f"[yellow]skipped[/yellow] {model_id}: no native RLE segmentation "
            f"predictions for response_mask_format={RESPONSE_MASK_FORMAT!r}"
        )
        return None
    return result


def median_seconds(fn: Callable[[], object], reps: int, warmup: int) -> float:
    """Return median runtime for ``fn``."""
    for _ in range(warmup):
        fn()
    gc.collect()

    timings = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def peak_bytes(fn: Callable[[], object]) -> int:
    """Return peak traced allocations for one call."""
    gc.collect()
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return int(peak)


def dense_mask_bytes(detections: sv.Detections) -> int:
    """Return dense mask storage bytes."""
    return 0 if detections.mask is None else int(np.asarray(detections.mask).nbytes)


def compact_mask_bytes(detections: sv.Detections) -> int:
    """Return compact mask storage bytes."""
    if not isinstance(detections.mask, CompactMask):
        return 0
    return sum(rle.nbytes for rle in detections.mask._rles)


def _fmt_ratio(ratio: float) -> str:
    """Format a speedup/compression ratio with colour coding."""
    fmt = f"{ratio:.0f}x" if ratio >= 10 else f"{ratio:.2f}x"
    if ratio >= 10:
        return f"[green]{fmt}[/green]"
    elif ratio >= 1:
        return f"[yellow]{fmt}[/yellow]"
    else:
        return f"[red]{fmt}[/red]"


def _fmt_mb(num_bytes: int) -> str:
    """Format bytes as compact megabytes."""
    return f"{num_bytes / 1e6:.2f}"


def run_benchmark(
    source: str,
    image: np.ndarray,
    result: dict[str, Any],
    reps: int,
    warmup: int,
) -> ApiBenchmarkResult:
    """Run one dense-vs-compact parser benchmark."""

    # Benchmark the public Roboflow/Inference adapter; RLE masks enter through
    # the result payload and should stay compact when compact_masks=True.
    def dense() -> sv.Detections:
        return sv.Detections.from_inference(result)

    def compact() -> sv.Detections:
        return sv.Detections.from_inference(result, compact_masks=True)

    dense_once = dense()
    compact_once = compact()
    if not isinstance(dense_once.mask, np.ndarray):
        raise TypeError(f"Expected dense ndarray mask, got {type(dense_once.mask)}")
    if not isinstance(compact_once.mask, CompactMask):
        raise TypeError(f"Expected CompactMask, got {type(compact_once.mask)}")
    np.testing.assert_array_equal(compact_once.mask.to_dense(), dense_once.mask)

    dense_s = median_seconds(dense, reps, warmup)
    compact_s = median_seconds(compact, reps, warmup)
    dense_peak = peak_bytes(dense)
    compact_peak = peak_bytes(compact)

    return ApiBenchmarkResult(
        source=source,
        resolution=f"{image.shape[1]}x{image.shape[0]}",
        segmented_objects=len(dense_once),
        dense_s=dense_s,
        compact_s=compact_s,
        dense_peak_bytes=dense_peak,
        compact_peak_bytes=compact_peak,
        dense_mask_bytes=dense_mask_bytes(dense_once),
        compact_mask_bytes=compact_mask_bytes(compact_once),
        pixel_perfect=True,
    )


def print_summary(results: list[ApiBenchmarkResult], reps: int, warmup: int) -> None:
    """Print a Rich summary table matching the compact mask benchmark style."""
    table = Table(
        title="CompactMask from_inference",
        box=box.ROUNDED,
        show_lines=False,
        header_style="bold cyan",
    )
    table.add_column("src", style="bold", no_wrap=True)
    table.add_column("res", no_wrap=True)
    table.add_column("seg", justify="right")
    table.add_column("dense ms", justify="right")
    table.add_column("CM ms", justify="right", style="green")
    table.add_column("speedup", justify="right")
    table.add_column("peak MB", justify="right", style="cyan")
    table.add_column("mask MB", justify="right")
    table.add_column("ok", justify="center")

    for result in results:
        speedup = result.dense_s / max(result.compact_s, 1e-9)
        table.add_row(
            result.source,
            result.resolution,
            str(result.segmented_objects),
            f"{result.dense_s * 1e3:.2f}",
            f"{result.compact_s * 1e3:.2f}",
            _fmt_ratio(speedup),
            f"{_fmt_mb(result.dense_peak_bytes)}/{_fmt_mb(result.compact_peak_bytes)}",
            f"{_fmt_mb(result.dense_mask_bytes)}/{_fmt_mb(result.compact_mask_bytes)}",
            "[green]✓[/green]" if result.pixel_perfect else "[red]✗[/red]",
        )

    console.print(table)
    console.print(
        "[dim]"
        + "  ·  ".join(
            [
                f"timings are median of {reps} reps after {warmup} warmups",
                "peak MB and mask MB are dense/compact",
                "speedup = dense / compact parse time; gains are allocation-driven"
                " (avoiding the dense (N,H,W) bool-stack), not faster RLE decode",
                "compact RLE arithmetic is typically slower than the dense NumPy path"
                " — synthetic-dense-64 shows this adversarial regime (speedup < 1x)",
                "OK means compact.to_dense() exactly matches dense masks",
            ]
        )
        + "[/dim]"
    )


def main() -> None:
    """Run the benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", choices=ASSETS.keys(), default=None)
    parser.add_argument("--image", type=Path, default=None)
    args = parser.parse_args()

    assets = [args.asset] if args.asset is not None else list(ASSETS)
    if args.image is not None:
        assets = ["custom"]

    results = []
    if args.asset is None and args.image is None:
        image, source, inference_result = synthetic_dense_small_result()
        console.rule(f"[bold]{source}[/bold] | {image.shape[1]}x{image.shape[0]}")
        results.append(
            run_benchmark(
                source=source,
                image=image,
                result=inference_result,
                reps=REPETITIONS,
                warmup=WARMUP,
            )
        )
    model_id = os.getenv(MODEL_ID_ENV, MODEL_ID)
    model = load_inference_model(model_id=model_id, api_key=os.getenv(API_KEY_ENV))
    for asset in assets:
        image, source = load_image_from_asset(args.image, asset)
        console.rule(f"[bold]{source}[/bold] | {image.shape[1]}x{image.shape[0]}")
        inference_result = run_inference_once(
            image=image,
            model=model,
            model_id=model_id,
            confidence=CONFIDENCE,
            iou=IOU,
        )
        if inference_result is None:
            continue
        console.print(
            f"[dim]captured {count_rle_predictions(inference_result)} RLE masks "
            f"from {model_id}[/dim]"
        )
        artifact = save_segmentation_artifact(
            image=image,
            result=inference_result,
            source=source,
        )
        if artifact is not None:
            console.print(f"[dim]saved segmentation artifact: {artifact}[/dim]")
        results.append(
            run_benchmark(
                source=source,
                image=image,
                result=inference_result,
                reps=REPETITIONS,
                warmup=WARMUP,
            )
        )
    if not results:
        raise ValueError(f"Model {model_id!r} returned no segmentation masks.")
    print_summary(results, reps=REPETITIONS, warmup=WARMUP)


if __name__ == "__main__":
    main()
