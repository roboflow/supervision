"""Benchmark mask painting strategies used by compact mask annotation.

Run with:
    uv run python examples/compact_mask/bench_mask_paint.py

This benchmark isolates the low-level painting work behind ``MaskAnnotator`` and
compares three strategies:

* full dense: paint from full-frame ``(N, H, W)`` boolean masks
* crop dense: decode each ``CompactMask`` crop, then paint the crop
* direct RLE: paint directly from ``CompactMask`` RLE true spans

The direct-RLE path exists to remove the last dense allocation from compact mask
painting. It should minimize transient memory and keep annotation viable for
large frames or high detection counts where full dense masks are impractical.
It is not expected to be universally faster than crop-dense painting: direct RLE
performs one Python slice assignment per true span, while crop-dense pays for a
small crop allocation and then lets NumPy's optimized boolean indexing do the
paint. The ``runs/spans`` column is therefore part of the benchmark output: many
short spans can make direct RLE slower in wall time even when it wins on memory.

The benchmark downloads supervision assets, runs one segmentation inference call
per image, and uses the resulting masks — matching the data source used by
``bench_inference_api.py``.
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

console = Console(width=160, force_terminal=True)

# Inference settings — keep in sync with bench_inference_api.py.
MODEL_ID = "rfdetr-seg-large"
MODEL_ID_ENV = "BENCH_INFERENCE_MODEL_ID"
API_KEY_ENV = "ROBOFLOW_API_KEY"
CONFIDENCE = 0.2
IOU = 0.5
RESPONSE_MASK_FORMAT = "rle"

REPETITIONS = 20
WARMUP = 3
MAX_OBJECTS_PER_SCENE = 120
COLORS_BGR = np.array(
    [
        [244, 67, 54],
        [33, 150, 243],
        [76, 175, 80],
        [255, 193, 7],
        [156, 39, 176],
        [255, 87, 34],
        [0, 188, 212],
        [139, 195, 74],
    ],
    dtype=np.uint8,
)

ASSETS = {Path(asset.filename).stem: asset for asset in ImageAssets}
for _video_asset in VideoAssets:
    _key = Path(_video_asset.filename).stem
    ASSETS[_key if _key not in ASSETS else f"{_key}-video"] = _video_asset


@dataclass(frozen=True, slots=True)
class PaintInput:
    """Prepared mask-painting input."""

    name: str
    scene: np.ndarray
    masks: np.ndarray
    xyxy: np.ndarray


@dataclass(frozen=True, slots=True)
class PaintBenchmarkResult:
    """Result for one mask-painting benchmark scenario."""

    scenario: str
    resolution: str
    objects: int
    mask_area_pct: float
    dense_storage_bytes: int
    compact_storage_bytes: int
    full_dense_s: float
    crop_dense_s: float
    direct_rle_s: float
    full_dense_peak_bytes: int
    crop_dense_peak_bytes: int
    direct_rle_peak_bytes: int
    rle_runs: int
    rle_spans: int
    crop_matches_dense: bool
    rle_matches_dense: bool


# ---------------------------------------------------------------------------
# Inference-based input generation  (mirrors bench_inference_api.py)
# ---------------------------------------------------------------------------


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


def load_inference_model(model_id: str, api_key: str | None) -> Any:
    """Load the requested Inference segmentation model."""
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
    """Run one segmentation inference call and return a frozen result dict."""
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
    if count_rle_predictions(result) == 0:
        console.print(
            f"[yellow]skipped[/yellow] {model_id}: no native RLE segmentation "
            f"predictions for response_mask_format={RESPONSE_MASK_FORMAT!r}"
        )
        return None
    return result


def make_inference_input(
    image: np.ndarray,
    result: dict[str, Any],
    name: str,
    max_objects: int,
) -> PaintInput:
    """Create a :class:`PaintInput` from a real segmentation inference result."""
    detections = sv.Detections.from_inference(result)
    if detections.mask is None or len(detections) == 0:
        raise ValueError(f"{name}: inference result contains no segmentation masks")
    masks_arr = np.asarray(detections.mask, dtype=bool)[:max_objects]
    xyxy = detections.xyxy[: len(masks_arr)].astype(np.float32)
    return PaintInput(name, image, masks_arr, xyxy)


def inference_inputs(
    assets: list[str],
    model: Any,
    model_id: str,
    confidence: float,
    iou: float,
    max_objects: int,
) -> list[PaintInput]:
    """Download supervision assets, run inference, return :class:`PaintInput` list."""
    inputs: list[PaintInput] = []
    for i, asset in enumerate(assets, 1):
        console.print(f"  [{i}/{len(assets)}] {asset}")
        try:
            image, _source = load_image_from_asset(None, asset)
            result = run_inference_once(image, model, model_id, confidence, iou)
            if result is not None:
                inputs.append(make_inference_input(image, result, asset, max_objects))
        except (FileNotFoundError, ValueError, OSError) as exc:
            console.print(f"[yellow]skipped[/yellow] {asset}: {exc}")
    return inputs


# ---------------------------------------------------------------------------
# Benchmark internals
# ---------------------------------------------------------------------------


def compact_storage_bytes(compact_mask: CompactMask) -> int:
    """Return raw bytes used by CompactMask's array-backed storage."""
    return int(
        compact_mask._crop_shapes.nbytes
        + compact_mask._offsets.nbytes
        + sum(rle.nbytes for rle in compact_mask._rles)
    )


def rle_run_count(compact_mask: CompactMask) -> int:
    """Return total RLE run count across all masks."""
    return int(sum(len(rle) for rle in compact_mask._rles))


def rle_span_count(compact_mask: CompactMask) -> int:
    """Return total direct-paint true-span count across all masks."""
    return int(
        sum(
            1
            for detection_idx in range(len(compact_mask))
            for _ in compact_mask._iter_true_spans(detection_idx)
        )
    )


def color_for_index(index: int) -> tuple[int, int, int]:
    """Return deterministic BGR color for an object index."""
    color = COLORS_BGR[index % len(COLORS_BGR)]
    return int(color[0]), int(color[1]), int(color[2])


def paint_full_dense_into(
    canvas: np.ndarray,
    masks: np.ndarray,
    order: np.ndarray,
) -> None:
    """Paint full-frame dense masks by boolean indexing into ``canvas``."""
    for detection_idx in order:
        canvas[masks[detection_idx]] = color_for_index(int(detection_idx))


def paint_crop_dense_into(
    canvas: np.ndarray,
    compact_mask: CompactMask,
    order: np.ndarray,
) -> None:
    """Paint CompactMask by decoding each crop to a dense boolean array."""
    for detection_idx in order:
        x1 = int(compact_mask.offsets[detection_idx, 0])
        y1 = int(compact_mask.offsets[detection_idx, 1])
        crop = compact_mask.crop(int(detection_idx))
        crop_h, crop_w = crop.shape
        canvas_slice = canvas[y1 : y1 + crop_h, x1 : x1 + crop_w]
        canvas_slice[crop] = color_for_index(int(detection_idx))


def paint_direct_rle_into(
    canvas: np.ndarray,
    compact_mask: CompactMask,
    order: np.ndarray,
) -> None:
    """Paint CompactMask via CompactMask.paint_into (batched RLE span scatter)."""
    for detection_idx in order:
        compact_mask.paint_into(
            canvas, int(detection_idx), color_for_index(int(detection_idx))
        )


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


def run_input(
    paint_input: PaintInput,
    repetitions: int,
    warmup: int,
) -> PaintBenchmarkResult:
    """Run all painting strategies for one prepared input."""
    scene = paint_input.scene
    masks = paint_input.masks
    compact_mask = CompactMask.from_dense(masks, paint_input.xyxy, scene.shape[:2])
    order = np.flip(np.argsort(compact_mask.area))
    full_canvas = np.empty_like(scene)
    crop_canvas = np.empty_like(scene)
    rle_canvas = np.empty_like(scene)

    def full_dense_inplace() -> np.ndarray:
        full_canvas[...] = scene
        paint_full_dense_into(full_canvas, masks, order)
        return full_canvas

    def crop_dense_inplace() -> np.ndarray:
        crop_canvas[...] = scene
        paint_crop_dense_into(crop_canvas, compact_mask, order)
        return crop_canvas

    def direct_rle_inplace() -> np.ndarray:
        rle_canvas[...] = scene
        paint_direct_rle_into(rle_canvas, compact_mask, order)
        return rle_canvas

    dense_result = scene.copy()
    paint_full_dense_into(dense_result, masks, order)
    crop_result = scene.copy()
    paint_crop_dense_into(crop_result, compact_mask, order)
    rle_result = scene.copy()
    paint_direct_rle_into(rle_result, compact_mask, order)

    image_h, image_w = scene.shape[:2]
    n_pixels = (
        max(1, masks.shape[1] * masks.shape[2])
        if masks.ndim == 3
        else max(1, image_h * image_w)
    )
    return PaintBenchmarkResult(
        scenario=paint_input.name,
        resolution=f"{image_w}x{image_h}",
        objects=len(masks),
        mask_area_pct=float(masks.sum() / (n_pixels * max(1, len(masks))) * 100),
        dense_storage_bytes=int(masks.nbytes),
        compact_storage_bytes=compact_storage_bytes(compact_mask),
        full_dense_s=median_seconds(full_dense_inplace, repetitions, warmup),
        crop_dense_s=median_seconds(crop_dense_inplace, repetitions, warmup),
        direct_rle_s=median_seconds(direct_rle_inplace, repetitions, warmup),
        full_dense_peak_bytes=peak_bytes(full_dense_inplace),
        crop_dense_peak_bytes=peak_bytes(crop_dense_inplace),
        direct_rle_peak_bytes=peak_bytes(direct_rle_inplace),
        rle_runs=rle_run_count(compact_mask),
        rle_spans=rle_span_count(compact_mask),
        crop_matches_dense=bool(np.array_equal(crop_result, dense_result)),
        rle_matches_dense=bool(np.array_equal(rle_result, dense_result)),
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _fmt_ratio(ratio: float) -> str:
    """Format a speedup/slowdown ratio with colour coding."""
    fmt = f"{ratio:.0f}x" if ratio >= 10 else f"{ratio:.2f}x"
    if ratio >= 1:
        return f"[green]{fmt}[/green]" if ratio >= 10 else f"[yellow]{fmt}[/yellow]"
    return f"[red]{fmt}[/red]"


def _fmt_mb(num_bytes: int) -> str:
    """Format bytes as compact megabytes."""
    return f"{num_bytes / 1e6:.2f}"


def print_summary(
    results: list[PaintBenchmarkResult],
    reps: int,
    warmup: int,
) -> None:
    """Print a Rich summary table."""
    table = Table(
        title="CompactMask mask painting",
        box=box.ROUNDED,
        show_lines=False,
        header_style="bold cyan",
    )
    table.add_column("src", style="bold", no_wrap=True)
    table.add_column("res", no_wrap=True)
    table.add_column("seg", justify="right")
    table.add_column("area%/obj", justify="right")
    table.add_column("full ms", justify="right")
    table.add_column("crop ms", justify="right")
    table.add_column("RLE ms", justify="right", style="green")
    table.add_column("RLE/full", justify="right")
    table.add_column("RLE/crop", justify="right")
    table.add_column("runs/spans", justify="right")
    table.add_column("paint MB", justify="right", style="cyan")
    table.add_column("mask MB", justify="right")
    table.add_column("ok", justify="center")

    for result in results:
        rle_full = result.full_dense_s / max(result.direct_rle_s, 1e-9)
        rle_crop = result.crop_dense_s / max(result.direct_rle_s, 1e-9)
        pixel_perfect = result.crop_matches_dense and result.rle_matches_dense
        table.add_row(
            result.scenario,
            result.resolution,
            str(result.objects),
            f"{result.mask_area_pct:.2f}",
            f"{result.full_dense_s * 1e3:.2f}",
            f"{result.crop_dense_s * 1e3:.2f}",
            f"{result.direct_rle_s * 1e3:.2f}",
            _fmt_ratio(rle_full),
            _fmt_ratio(rle_crop),
            f"{result.rle_runs}/{result.rle_spans}",
            f"{_fmt_mb(result.crop_dense_peak_bytes)}/{_fmt_mb(result.direct_rle_peak_bytes)}",
            f"{_fmt_mb(result.dense_storage_bytes)}/{_fmt_mb(result.compact_storage_bytes)}",
            "[green]✓[/green]" if pixel_perfect else "[red]✗[/red]",
        )
    console.print(table)
    console.print(
        "[dim]"
        + "  ·  ".join(
            [
                f"timings are median of {reps} reps after {warmup} warmups",
                "paint MB is crop/RLE peak traced allocation while painting "
                "into a preallocated canvas",
                "mask MB is dense/compact persistent mask storage",
                "RLE/full = full dense paint time / direct RLE paint time",
                "RLE/crop = crop-dense paint time / direct RLE paint time",
                "runs/spans exposes Python direct-paint loop count; many spans "
                "can be slower than crop-dense NumPy indexing",
                "area%/obj = per-object mask coverage as % of frame area",
                "OK means crop-dense and direct-RLE outputs exactly match full dense",
            ]
        )
        + "[/dim]"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", choices=ASSETS.keys(), default=None)
    parser.add_argument("--image", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Run the benchmark."""
    args = parse_args()
    model_id = os.getenv(MODEL_ID_ENV, MODEL_ID)
    api_key = os.getenv(API_KEY_ENV)
    model = load_inference_model(model_id, api_key)
    assets = [args.asset] if args.asset is not None else list(ASSETS)
    if args.image is not None:
        assets = ["custom"]
    inputs = inference_inputs(
        assets=assets,
        model=model,
        model_id=model_id,
        confidence=CONFIDENCE,
        iou=IOU,
        max_objects=MAX_OBJECTS_PER_SCENE,
    )
    if not inputs:
        console.print("[yellow]no inference inputs found; exiting[/yellow]")
        return

    results: list[PaintBenchmarkResult] = []
    for i, paint_input in enumerate(inputs, 1):
        console.print(f"  [{i}/{len(inputs)}] {paint_input.name}")
        results.append(
            run_input(
                paint_input=paint_input,
                repetitions=REPETITIONS,
                warmup=WARMUP,
            )
        )
        gc.collect()

    print_summary(results, reps=REPETITIONS, warmup=WARMUP)


if __name__ == "__main__":
    main()
