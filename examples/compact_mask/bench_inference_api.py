"""Benchmark dense vs compact Roboflow RLE ingestion.

Run with:
    uv run python examples/compact_mask/bench_inference_api.py

The benchmark downloads a supervision image asset, then builds
RF-DETR-like segmentation predictions encoded as Roboflow ``rle_mask`` payloads.
No model download is required.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc
import zlib
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

REPETITIONS = 20
WARMUP = 3
FHD_AREA = 1920 * 1080
FHD_OBJECTS = 16

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


def image_shape_from_asset(path: Path | None, asset: str) -> tuple[int, int, str]:
    """Return ``(height, width, label)`` for an image or video middle frame."""
    if path is not None:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return int(image.shape[0]), int(image.shape[1]), str(path)

    asset_obj = ASSETS[asset]
    asset_path = Path(download_assets(asset_obj))
    if isinstance(asset_obj, ImageAssets):
        image = cv2.imread(str(asset_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {asset_path}")
        return int(image.shape[0]), int(image.shape[1]), str(asset_path)

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
    return int(frame.shape[0]), int(frame.shape[1]), f"{asset_path}#{frame_index}"


def make_rfdetr_like_result(
    image_height: int,
    image_width: int,
    object_count: int,
) -> dict[str, Any]:
    """Build a Roboflow result with person-like RLE segmentation predictions."""
    rng = np.random.default_rng(0)
    cols = max(1, int(np.ceil(np.sqrt(object_count * image_width / image_height))))
    rows = max(1, int(np.ceil(object_count / cols)))
    cell_w = image_width / cols
    cell_h = image_height / rows
    predictions: list[dict[str, Any]] = []

    for index in range(object_count):
        col = index % cols
        row = index // cols
        cx = int((col + 0.5) * cell_w + rng.integers(-8, 9))
        cy = int((row + 0.5) * cell_h + rng.integers(-8, 9))
        box_w = max(16, int(cell_w * 0.34))
        box_h = max(32, int(cell_h * 0.58))
        x1 = max(0, cx - box_w // 2)
        y1 = max(0, cy - box_h // 2)
        x2 = min(image_width - 1, cx + box_w // 2)
        y2 = min(image_height - 1, cy + box_h // 2)

        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = (max(4, (x2 - x1) // 3), max(8, (y2 - y1) // 3))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)

        predictions.append(
            {
                "x": (x1 + x2) / 2,
                "y": (y1 + y2) / 2,
                "width": x2 - x1,
                "height": y2 - y1,
                "confidence": 0.9,
                "class_id": 0,
                "class": "person",
                "rle_mask": {
                    "size": [image_height, image_width],
                    "counts": sv.mask_to_rle(mask.astype(bool), compressed=True),
                },
            }
        )

    return {
        "image": {"width": image_width, "height": image_height},
        "predictions": predictions,
    }


def estimate_object_count(
    source: str,
    image_height: int,
    image_width: int,
) -> int:
    """Estimate deterministic synthetic detections from source and image size."""
    area_scale = image_height * image_width / FHD_AREA
    base_count = max(1, round(FHD_OBJECTS * area_scale))
    source_scale = 0.65 + (zlib.crc32(source.encode()) % 36) / 100
    return max(1, round(base_count * source_scale))


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
    image_height: int,
    image_width: int,
    reps: int,
    warmup: int,
) -> ApiBenchmarkResult:
    """Run one dense-vs-compact parser benchmark."""
    synthetic_objects = estimate_object_count(source, image_height, image_width)
    result = make_rfdetr_like_result(image_height, image_width, synthetic_objects)

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
        resolution=f"{image_width}x{image_height}",
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
    parser.add_argument("--reps", type=int, default=REPETITIONS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    args = parser.parse_args()

    assets = [args.asset] if args.asset is not None else list(ASSETS)
    if args.image is not None:
        assets = ["custom"]

    results = []
    for asset in assets:
        image_height, image_width, source = image_shape_from_asset(args.image, asset)
        console.rule(f"[bold]{source}[/bold] | {image_width}x{image_height}")
        results.append(
            run_benchmark(
                source=source,
                image_height=image_height,
                image_width=image_width,
                reps=args.reps,
                warmup=args.warmup,
            )
        )
    print_summary(results, reps=args.reps, warmup=args.warmup)


if __name__ == "__main__":
    main()
