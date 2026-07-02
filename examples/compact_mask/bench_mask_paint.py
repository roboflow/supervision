"""Benchmark sv.MaskAnnotator painting strategies for CompactMask inputs.

Run with:
    uv run python examples/compact_mask/bench_mask_paint.py

Compares three sv.MaskAnnotator calling conventions on real segmentation data:

* full_dense  — standard dense (N, H, W) bool masks; pre-CompactMask baseline
* direct_rle  — CompactMask with compact_mask_strategy="direct_rle" (default)
* crop_dense  — CompactMask with compact_mask_strategy="crop_dense"

Timings cover the full sv.MaskAnnotator.annotate() call including opacity
blending, matching what users observe in production. To switch strategy:

    annotator = sv.MaskAnnotator(compact_mask_strategy="crop_dense")

The benchmark downloads supervision assets, runs one segmentation inference
call per image, and uses the resulting masks.
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

from supervision import Detections, MaskAnnotator, mask_to_xyxy, rle_to_mask
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

ASSETS = {Path(asset.filename).stem: asset for asset in ImageAssets}
for _video_asset in VideoAssets:
    _key = Path(_video_asset.filename).stem
    ASSETS[_key if _key not in ASSETS else f"{_key}-video"] = _video_asset


@dataclass(frozen=True, slots=True)
class PaintInput:
    """Prepared mask-painting input."""

    name: str
    scene: np.ndarray
    masks: np.ndarray  # (N, H, W) bool — dense
    xyxy: np.ndarray  # (N, 4) float32
    class_id: np.ndarray  # (N,) int — for consistent color lookup across strategies


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
    direct_rle_s: float
    crop_dense_s: float
    full_dense_peak_bytes: int
    direct_rle_peak_bytes: int
    crop_dense_peak_bytes: int
    rle_runs: int
    rle_spans: int
    rle_matches_full: bool
    crop_matches_full: bool


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
        mask = rle_to_mask(rle["counts"], resolution_wh=(int(width), int(height)))
        if not mask.any():
            predictions.append(prediction)
            continue

        x1, y1, x2, y2 = mask_to_xyxy(mask[np.newaxis, ...])[0]
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
    detections = Detections.from_inference(result)
    if detections.mask is None or len(detections) == 0:
        raise ValueError(f"{name}: inference result contains no segmentation masks")
    n = min(max_objects, len(detections))
    masks_arr = np.asarray(detections.mask, dtype=bool)[:n]
    xyxy = detections.xyxy[:n].astype(np.float32)
    class_id = (
        detections.class_id[:n]
        if detections.class_id is not None
        else np.zeros(n, dtype=np.int_)
    )
    return PaintInput(name, image, masks_arr, xyxy, class_id)


def infer_and_paint(
    asset: str,
    image_path: Path | None,
    model: Any,
    model_id: str,
    confidence: float,
    iou: float,
    max_objects: int,
    repetitions: int,
    warmup: int,
) -> PaintBenchmarkResult | None:
    """Load image, run inference, run paint benchmark — one image end-to-end."""
    try:
        image, _source = load_image_from_asset(image_path, asset)
    except (FileNotFoundError, OSError) as exc:
        console.print(f"  [yellow]skipped[/yellow] {asset}: {exc}")
        return None
    result = run_inference_once(image, model, model_id, confidence, iou)
    if result is None:
        return None
    try:
        paint_input = make_inference_input(image, result, asset, max_objects)
    except ValueError as exc:
        console.print(f"  [yellow]skipped[/yellow] {asset}: {exc}")
        return None
    return run_input(paint_input, repetitions=repetitions, warmup=warmup)


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
    """Run all three MaskAnnotator strategies for one prepared input."""
    scene = paint_input.scene
    masks = paint_input.masks
    class_id = paint_input.class_id
    compact_mask = CompactMask.from_dense(masks, paint_input.xyxy, scene.shape[:2])

    # Dense detections use the old (N, H, W) bool mask path in MaskAnnotator.
    det_dense = Detections(xyxy=paint_input.xyxy, mask=masks, class_id=class_id)
    # Compact detections route through CompactMask; strategy selects the sub-path.
    det_compact = Detections(
        xyxy=paint_input.xyxy, mask=compact_mask, class_id=class_id
    )

    ann_full = MaskAnnotator()
    ann_rle = MaskAnnotator(compact_mask_strategy="direct_rle")
    ann_crop = MaskAnnotator(compact_mask_strategy="crop_dense")

    def full_dense() -> np.ndarray:
        return ann_full.annotate(scene=scene.copy(), detections=det_dense)

    def direct_rle() -> np.ndarray:
        return ann_rle.annotate(scene=scene.copy(), detections=det_compact)

    def crop_dense() -> np.ndarray:
        return ann_crop.annotate(scene=scene.copy(), detections=det_compact)

    full_result = full_dense()
    rle_result = direct_rle()
    crop_result = crop_dense()

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
        full_dense_s=median_seconds(full_dense, repetitions, warmup),
        direct_rle_s=median_seconds(direct_rle, repetitions, warmup),
        crop_dense_s=median_seconds(crop_dense, repetitions, warmup),
        full_dense_peak_bytes=peak_bytes(full_dense),
        direct_rle_peak_bytes=peak_bytes(direct_rle),
        crop_dense_peak_bytes=peak_bytes(crop_dense),
        rle_runs=rle_run_count(compact_mask),
        rle_spans=rle_span_count(compact_mask),
        rle_matches_full=bool(np.array_equal(rle_result, full_result)),
        crop_matches_full=bool(np.array_equal(crop_result, full_result)),
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _fmt_ratio(ratio: float) -> str:
    """Format crop/RLE ratio: >1 = direct_rle faster (green), <1 = crop faster (red)."""
    fmt = f"{ratio:.0f}x" if ratio >= 10 else f"{ratio:.2f}x"
    if ratio >= 1:
        return f"[green]{fmt}[/green]" if ratio >= 10 else f"[yellow]{fmt}[/yellow]"
    return f"[red]{fmt}[/red]"


def _fmt_ms(value_s: float, winner: bool) -> str:
    """Format milliseconds, green when this strategy is faster."""
    s = f"{value_s * 1e3:.2f}"
    return f"[green]{s}[/green]" if winner else s


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
        title="sv.MaskAnnotator painting strategies",
        box=box.ROUNDED,
        show_lines=False,
        header_style="bold cyan",
    )
    table.add_column("source / image", style="bold", no_wrap=True)
    table.add_column("resolution", no_wrap=True)
    table.add_column("seg", justify="right")
    table.add_column("area%/obj", justify="right")
    table.add_column("direct_rle ms", justify="right")
    table.add_column("crop_dense ms", justify="right")
    table.add_column("crop/RLE", justify="right")
    table.add_column("runs/spans", justify="right")
    table.add_column("annot MB", justify="right", style="cyan")
    table.add_column("mask MB", justify="right")
    table.add_column("ok", justify="center")

    for result in results:
        rle_faster = result.direct_rle_s <= result.crop_dense_s
        crop_vs_rle = result.crop_dense_s / max(result.direct_rle_s, 1e-9)
        annot_ratio = result.direct_rle_peak_bytes / max(
            result.crop_dense_peak_bytes, 1
        )
        mask_ratio = result.dense_storage_bytes / max(result.compact_storage_bytes, 1)
        pixel_perfect = result.rle_matches_full and result.crop_matches_full
        table.add_row(
            result.scenario,
            result.resolution,
            str(result.objects),
            f"{result.mask_area_pct:.2f}",
            _fmt_ms(result.direct_rle_s, winner=rle_faster),
            _fmt_ms(result.crop_dense_s, winner=not rle_faster),
            _fmt_ratio(crop_vs_rle),
            f"{result.rle_runs / max(result.rle_spans, 1):.2f}",
            f"{annot_ratio:.2f}",
            f"{mask_ratio:.0f}x",
            "[green]✓[/green]" if pixel_perfect else "[red]✗[/red]",
        )
    console.print(table)
    console.print(
        "[dim]"
        + "  ·  ".join(
            [
                f"timings are median of {reps} reps after {warmup} warmups",
                "timings = full sv.MaskAnnotator.annotate() including opacity blend",
                "both strategies use CompactMask; green ms = faster strategy per scene",
                "crop/RLE = crop_dense_ms / direct_rle_ms"
                " (>1 = direct_rle faster, <1 = crop_dense faster)",
                "annot MB = direct_rle/crop_dense peak traced bytes ratio",
                "mask MB = dense/compact storage ratio (higher = more savings)",
                "runs/spans = RLE runs / true-pixel column spans ratio"
                " (higher = more fragmented mask)",
                "area%/obj = per-object mask coverage as % of frame area",
                "OK means direct_rle and crop_dense outputs exactly match full_dense",
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

    results: list[PaintBenchmarkResult] = []
    for i, asset in enumerate(assets, 1):
        console.print(f"[{i}/{len(assets)}] {asset}")
        result = infer_and_paint(
            asset=asset,
            image_path=args.image,
            model=model,
            model_id=model_id,
            confidence=CONFIDENCE,
            iou=IOU,
            max_objects=MAX_OBJECTS_PER_SCENE,
            repetitions=REPETITIONS,
            warmup=WARMUP,
        )
        if result is not None:
            results.append(result)
        gc.collect()

    print_summary(results, reps=REPETITIONS, warmup=WARMUP)


if __name__ == "__main__":
    main()
