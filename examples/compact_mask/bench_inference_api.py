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
from supervision.detection.compact_mask import CompactMask

console = Console(width=120, force_terminal=True)

MODEL_ID = "yolov8l-seg-640"
MODEL_ID_ENV = "BENCH_INFERENCE_MODEL_ID"
API_KEY_ENV = "ROBOFLOW_API_KEY"
CONFIDENCE = 0.3
IOU = 0.5
REPETITIONS = 20
WARMUP = 3

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


def _prediction_points_to_polygon(prediction: dict[str, Any]) -> np.ndarray | None:
    """Return polygon coordinates from a Roboflow segmentation prediction."""
    points = prediction.get("points")
    if not points:
        return None
    polygon = np.array(
        [
            [point["x"], point["y"]] if isinstance(point, dict) else [point.x, point.y]
            for point in points
        ],
        dtype=np.int32,
    )
    return polygon if len(polygon) >= 3 else None


def normalize_to_rle_masks(result: dict[str, Any]) -> dict[str, Any]:
    """Ensure real segmentation predictions carry Roboflow ``rle_mask`` payloads."""
    if count_rle_predictions(result) > 0:
        return result

    image = result["image"]
    image_width = int(image["width"])
    image_height = int(image["height"])
    predictions = []
    for prediction in result.get("predictions", []):
        polygon = _prediction_points_to_polygon(prediction)
        if polygon is None:
            predictions.append(prediction)
            continue
        mask = sv.polygon_to_mask(
            polygon=polygon, resolution_wh=(image_width, image_height)
        ).astype(bool)
        if mask.any():
            x1, y1, x2, y2 = sv.mask_to_xyxy(mask[np.newaxis, ...])[0]
            prediction = {
                **prediction,
                "x": float((x1 + x2) / 2),
                "y": float((y1 + y2) / 2),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
            }
        predictions.append(
            {
                **prediction,
                "rle_mask": {
                    "size": [image_height, image_width],
                    "counts": sv.mask_to_rle(mask, compressed=True),
                },
            }
        )
    return {**result, "predictions": predictions}


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
    result = normalize_to_rle_masks(
        freeze_result(model.infer(image, confidence=confidence, iou=iou)[0])
    )
    rle_count = count_rle_predictions(result)
    if rle_count == 0:
        console.print(
            f"[yellow]skipped[/yellow] {model_id}: no RLE or polygon segmentation "
            "predictions"
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
    table.add_column("x", justify="right")
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
