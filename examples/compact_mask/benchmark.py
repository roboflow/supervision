"""CompactMask demo & benchmark.

Demonstrates that ``CompactMask`` is a drop-in replacement for dense
``(N, H, W)`` bool arrays in ``supervision.Detections``, while using
significantly less memory and enabling faster annotation.

Run with:
    uv run python examples/compact_mask/benchmark.py

No GPU or real model is required — everything is synthesized with NumPy.
Mask complexity is controlled by ``num_vertices``: random polygons with more
vertices produce jaggier boundaries and more RLE runs per row.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import math
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

import supervision as sv
from supervision.detection.compact_mask import CompactMask

console = Console(width=240, force_terminal=True)

REPETITIONS = 4
# How many reps to run concurrently in time_reps. Each thread times itself
# independently; results are averaged. Numpy releases the GIL for its C-level
# work so threads can truly run in parallel on multi-core machines.
# Set to 1 to disable parallelism and revert to a sequential timing loop.
PARALLEL = 3
# Dense timing is skipped when the dense (N,H,W) array would exceed this
# threshold — avoids OOM / swap thrashing on extreme scenarios while still
# reporting the theoretical memory footprint.
DENSE_SKIP_GB = 16.0
# Dense IoU *and NMS* timing are skipped above this threshold: pairwise
# (N,H,W) AND is extremely expensive — NMS calls IoU internally so both are
# gated by the same threshold.
IOU_DENSE_SKIP_GB = 1.0
# Reps for dense IoU/NMS — a single pass already takes several seconds.
IOU_NMS_REPS = 2


# ══════════════════════════════════════════════════════════════════════════════
# Result container
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ScenarioResult:
    name: str
    resolution: str  # e.g. "1920x1080"
    num_objects: int
    fill_name: str  # e.g. "5%"
    num_vertices: int  # polygon vertex count — complexity proxy
    # memory (theoretical: raw numpy nbytes)
    dense_bytes: int
    compact_bytes_theoretical: int
    # memory (actual: tracemalloc peak; dense_bytes_actual=0 when dense_skipped=True)
    dense_bytes_actual: int
    compact_bytes_actual: int
    # compactness overhead — absolute times for conversion (always measured)
    encode_s: float  # CompactMask.from_dense()  dense → compact
    decode_s: float  # compact_mask.to_dense()   compact → dense
    # timing (nan when dense_skipped=True)
    dense_area_s: float
    compact_area_s: float
    dense_filter_s: float
    compact_filter_s: float
    dense_annot_s: float
    compact_annot_s: float
    # pipeline stages (nan when respective skip flag is True)
    dense_iou_s: float  # nan when iou_dense_skipped
    compact_iou_s: float
    dense_nms_s: float  # nan when dense_skipped
    compact_nms_s: float
    dense_merge_s: float  # nan when dense_skipped
    compact_merge_s: float
    dense_offset_s: float  # nan when dense_skipped
    compact_offset_s: float
    dense_centroids_s: float  # nan when dense_skipped
    compact_centroids_s: float
    # correctness (None when the stage was skipped)
    pixel_perfect: bool | None
    areas_match: bool | None
    roundtrip_ok: bool | None
    iou_ok: bool | None
    nms_ok: bool | None
    nms_mismatch_count: (
        int  # detections with different NMS decisions (0 when dense_skipped)
    )
    merge_ok: bool | None
    offset_ok: bool | None
    centroids_ok: bool | None
    # skip flags
    dense_skipped: bool = field(default=False)
    iou_dense_skipped: bool = field(default=False)


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic data helpers
# ══════════════════════════════════════════════════════════════════════════════


def make_scene(image_height: int, image_width: int) -> np.ndarray:
    """Random BGR image."""
    return np.random.default_rng(42).integers(
        0, 255, (image_height, image_width, 3), dtype=np.uint8
    )


def _make_polygon_mask(
    image_height: int,
    image_width: int,
    center_x: int,
    center_y: int,
    axis_x: int,
    axis_y: int,
    rng: np.random.Generator,
    num_vertices: int,
) -> np.ndarray:
    """Random polygon mask.

    *num_vertices* is a direct complexity proxy: more vertices → more
    independent radius samples → jaggier boundary → more RLE runs per row.
    No smoothing is applied so the relationship is monotone.
    """
    angles = np.sort(rng.uniform(0, 2 * np.pi, num_vertices))
    radii = rng.uniform(0.3, 1.0, num_vertices)
    pts_x = np.clip(
        (center_x + axis_x * radii * np.cos(angles)).astype(np.int32),
        0,
        image_width - 1,
    )
    pts_y = np.clip(
        (center_y + axis_y * radii * np.sin(angles)).astype(np.int32),
        0,
        image_height - 1,
    )
    pts = np.column_stack([pts_x, pts_y]).reshape(-1, 1, 2)
    canvas = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.fillPoly(canvas, [pts], 1)
    return canvas.astype(bool)


def make_detections(
    num_objects: int,
    image_height: int,
    image_width: int,
    fill_fraction: float,
    num_vertices: int = 20,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(xyxy, masks_dense, class_ids)`` with random polygon masks.

    *num_vertices* controls mask complexity: more vertices → jaggier boundary.
    """
    rng = np.random.default_rng(seed)
    half = max(
        2,
        int(
            (image_height * image_width * fill_fraction / (np.pi * num_objects)) ** 0.5
        ),
    )
    xyxy_list = []
    masks = np.zeros((num_objects, image_height, image_width), dtype=bool)
    for index in range(num_objects):
        center_x = int(rng.integers(half + 1, image_width - half - 1))
        center_y = int(rng.integers(half + 1, image_height - half - 1))
        axis_x = int(rng.integers(max(2, half // 2), half * 2 + 1))
        axis_y = int(rng.integers(max(2, half // 2), half * 2 + 1))
        masks[index] = _make_polygon_mask(
            image_height,
            image_width,
            center_x,
            center_y,
            axis_x,
            axis_y,
            rng,
            num_vertices,
        )
        xyxy_list.append(
            [
                max(0, center_x - axis_x),
                max(0, center_y - axis_y),
                min(image_width - 1, center_x + axis_x),
                min(image_height - 1, center_y + axis_y),
            ]
        )
    xyxy = np.array(xyxy_list, dtype=np.float32)
    class_ids = rng.integers(0, 10, num_objects, dtype=int)
    return xyxy, masks, class_ids


# ══════════════════════════════════════════════════════════════════════════════
# Memory helpers
# ══════════════════════════════════════════════════════════════════════════════


def dense_memory_bytes(masks: np.ndarray) -> int:
    """Theoretical dense footprint: raw numpy buffer size."""
    return int(masks.nbytes)


def compact_memory_bytes_theoretical(compact_mask: CompactMask) -> int:
    """Theoretical compact footprint: sum of all internal numpy buffer sizes."""
    return int(
        compact_mask._crop_shapes.nbytes
        + compact_mask._offsets.nbytes
        + sum(rle.nbytes for rle in compact_mask._rles),
    )


def measure_peak_bytes(func: Callable[[], object]) -> int:
    """Wrapper that runs *func* under tracemalloc and returns peak allocation.

    tracemalloc captures every Python-level allocation — numpy buffers, list
    nodes, object headers — giving the true heap cost of anything *func*
    builds. The return value of *func* is discarded so the object does not
    stay alive.
    """
    tracemalloc.start()
    tracemalloc.clear_traces()
    func()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return int(peak)


def dense_memory_bytes_actual(
    num_objects: int, image_height: int, image_width: int
) -> int:
    """Actual dense footprint: peak bytes during (N, H, W) bool array alloc."""
    return measure_peak_bytes(
        lambda: np.zeros((num_objects, image_height, image_width), dtype=bool),
    )


def compact_memory_bytes_actual(
    masks_dense: np.ndarray,
    xyxy: np.ndarray,
    image_shape: tuple[int, int],
) -> int:
    """Actual compact footprint: peak bytes during CompactMask.from_dense()."""
    return measure_peak_bytes(
        lambda: CompactMask.from_dense(masks_dense, xyxy, image_shape=image_shape),
    )


def time_reps(
    func: Callable[[], object],
    repeats: int = REPETITIONS,
    parallel: int = PARALLEL,
) -> float:
    """Run *func* *reps* times and return mean wall-clock seconds per call.

    When ``parallel > 1``, up to ``parallel`` calls run simultaneously in
    threads. Numpy and OpenCV release the GIL for their C-level work, so
    threads can execute in parallel on multi-core machines. Each thread
    records its own elapsed time; the mean across all *reps* is returned.

    When ``parallel == 1`` the original sequential loop is used, avoiding
    any thread-scheduling overhead and improving accuracy for cheap functions.

    A full GC cycle is run before timing so accumulated garbage from earlier
    stages does not trigger collection mid-measurement and inflate results.
    """
    gc.collect()
    if parallel <= 1:
        t0 = time.perf_counter()
        for _ in range(repeats):
            func()
        return (time.perf_counter() - t0) / repeats

    def _timed() -> float:
        t0 = time.perf_counter()
        func()
        return time.perf_counter() - t0

    with ThreadPoolExecutor(max_workers=min(parallel, repeats)) as pool:
        timings = list(pool.map(lambda _: _timed(), range(repeats)))
    return sum(timings) / repeats


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark stages
# ══════════════════════════════════════════════════════════════════════════════


def stage_build(
    num_objects: int,
    image_height: int,
    image_width: int,
    fill_fraction: float,
    num_vertices: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, CompactMask]:
    """Synthesize polygon masks and build the CompactMask."""
    xyxy, masks_dense, class_ids = make_detections(
        num_objects, image_height, image_width, fill_fraction, num_vertices
    )
    compact_mask = CompactMask.from_dense(
        masks_dense, xyxy, image_shape=(image_height, image_width)
    )
    return xyxy, masks_dense, class_ids, compact_mask


def stage_encode(
    masks_dense: np.ndarray,
    xyxy: np.ndarray,
    image_height: int,
    image_width: int,
) -> float:
    """Per-mask encode time: encode each mask individually and average over N.

    Calling from_dense one mask at a time (rather than batching all N) isolates
    the per-shape cost — each polygon has a different RLE run count, so the
    average reflects true shape variance.
    """
    num_masks = len(masks_dense)
    image_shape = (image_height, image_width)

    def _encode_each() -> None:
        for i in range(num_masks):
            CompactMask.from_dense(
                masks_dense[i : i + 1], xyxy[i : i + 1], image_shape=image_shape
            )

    return time_reps(_encode_each) / max(num_masks, 1)


def stage_decode(compact_mask: CompactMask) -> float:
    """Per-mask decode time: decode each mask individually and average over N.

    Building a list via compact_mask[i] decodes each crop separately, giving
    the per-mask cost of materialising a single RLE back to a dense array.
    """
    num_masks = len(compact_mask)
    return time_reps(lambda: [compact_mask[i] for i in range(num_masks)]) / max(
        num_masks, 1
    )


def stage_area(
    det_dense: sv.Detections, det_compact: sv.Detections
) -> tuple[float, float]:
    """Time .area on both representations."""
    return (
        time_reps(lambda: det_dense.area),
        time_reps(lambda: det_compact.area),
    )


def stage_filter(
    det_dense: sv.Detections, det_compact: sv.Detections
) -> tuple[float, float]:
    """Time boolean filtering (keep every other detection)."""
    keep = np.arange(len(det_dense)) % 2 == 0
    return (
        time_reps(lambda: det_dense[keep]),
        time_reps(lambda: det_compact[keep]),
    )


def stage_annotate(
    scene: np.ndarray, det_dense: sv.Detections, det_compact: sv.Detections
) -> tuple[float, float]:
    """Time MaskAnnotator on both representations."""
    annotator = sv.MaskAnnotator(opacity=0.5)
    return (
        time_reps(lambda: annotator.annotate(scene.copy(), det_dense)),
        time_reps(lambda: annotator.annotate(scene.copy(), det_compact)),
    )


def stage_correctness(
    scene: np.ndarray,
    masks_dense: np.ndarray,
    compact_mask: CompactMask,
    det_dense: sv.Detections,
    det_compact: sv.Detections,
) -> tuple[bool, bool, bool]:
    """Return (pixel_perfect, areas_match, roundtrip_ok)."""
    annotator = sv.MaskAnnotator(opacity=0.5)
    out_dense = annotator.annotate(scene.copy(), det_dense)
    out_compact = annotator.annotate(scene.copy(), det_compact)
    pixel_perfect = bool(np.array_equal(out_dense, out_compact))
    areas_match = bool(np.allclose(det_dense.area, det_compact.area))
    roundtrip_ok = bool(np.array_equal(compact_mask.to_dense(), masks_dense))
    return pixel_perfect, areas_match, roundtrip_ok


def stage_iou(
    masks_dense: np.ndarray,
    compact_mask: CompactMask,
    iou_dense_skipped: bool,
) -> tuple[float, float, bool | None]:
    """Time pairwise self-IoU using dense (N,H,W) AND and compact crop filter.

    Correctness is checked on the first 10 masks only to keep it fast,
    regardless of whether full dense IoU timing is skipped.
    """
    correct_n = min(len(compact_mask), 10)
    iou_compact_small = sv.mask_iou_batch(
        compact_mask[:correct_n], compact_mask[:correct_n]
    )
    iou_dense_small = sv.mask_iou_batch(
        masks_dense[:correct_n], masks_dense[:correct_n]
    )
    iou_ok = bool(np.allclose(iou_dense_small, iou_compact_small, atol=1e-4))

    compact_iou_s = time_reps(lambda: sv.mask_iou_batch(compact_mask, compact_mask))
    if iou_dense_skipped:
        dense_iou_s = math.nan
    else:
        dense_iou_s = time_reps(
            lambda: sv.mask_iou_batch(masks_dense, masks_dense),
            repeats=IOU_NMS_REPS,
        )
    return dense_iou_s, compact_iou_s, iou_ok


def stage_nms(
    xyxy: np.ndarray,
    confidence: np.ndarray,
    class_ids: np.ndarray,
    masks_dense: np.ndarray,
    compact_mask: CompactMask,
    dense_skipped: bool,
    iou_dense_skipped: bool,
) -> tuple[float, float, bool | None, int]:
    """Time mask NMS. Dense resizes to 640 before IoU; compact uses exact crop IoU.

    Compact NMS is strictly more accurate than dense: it computes pixel-level IoU
    directly on the full-resolution RLE crops instead of a lossy 640px-downsampled
    approximation.  For pairs whose true IoU is very close to the 0.5 threshold,
    the resize step in the dense path can flip a keep/suppress decision.

    ``n_diff`` counts detections whose decision differs between the two paths.
    ``nms_ok`` is True when ``n_diff == 0``.

    Dense NMS is skipped when ``dense_skipped`` *or* ``iou_dense_skipped`` is True:
    NMS calls mask_iou_batch internally so the cost is the same as IoU.

    Returns:
        Tuple of ``(dense_nms_s, compact_nms_s, nms_ok, n_diff)``.
    """
    predictions = np.c_[xyxy, confidence, class_ids.astype(float)]

    compact_nms_s = time_reps(
        lambda: sv.mask_non_max_suppression(predictions, compact_mask)
    )
    if dense_skipped or iou_dense_skipped:
        return math.nan, compact_nms_s, None, 0

    keep_dense = sv.mask_non_max_suppression(predictions, masks_dense)
    keep_compact = sv.mask_non_max_suppression(predictions, compact_mask)
    n_diff = int(np.sum(keep_dense != keep_compact))
    nms_ok = n_diff == 0
    dense_nms_s = time_reps(
        lambda: sv.mask_non_max_suppression(predictions, masks_dense),
        repeats=IOU_NMS_REPS,
    )
    return dense_nms_s, compact_nms_s, nms_ok, n_diff


def stage_merge(
    det_dense: sv.Detections | None,
    det_compact: sv.Detections,
    dense_skipped: bool,
) -> tuple[float, float, bool | None]:
    """Time Detections.merge on two half-splits.

    Dense: np.vstack; compact: RLE concat.
    Splits are pre-computed so the timed lambda measures only the merge.
    """
    half = len(det_compact) // 2
    compact_a, compact_b = det_compact[:half], det_compact[half:]

    compact_merge_s = time_reps(lambda: sv.Detections.merge([compact_a, compact_b]))
    if dense_skipped or det_dense is None:
        return math.nan, compact_merge_s, None

    dense_a, dense_b = det_dense[:half], det_dense[half:]
    merged_d = sv.Detections.merge([dense_a, dense_b])
    merged_c = sv.Detections.merge([compact_a, compact_b])
    merge_ok = bool(np.allclose(merged_d.area, merged_c.area))
    dense_merge_s = time_reps(lambda: sv.Detections.merge([dense_a, dense_b]))
    return dense_merge_s, compact_merge_s, merge_ok


def stage_offset(
    masks_dense: np.ndarray,
    compact_mask: CompactMask,
    image_height: int,
    image_width: int,
    dense_skipped: bool,
) -> tuple[float, float, bool | None]:
    """Time mask offset: move_masks (N,H,W) copy vs O(N) offset update."""
    dx, dy = 10, 10
    # Expand the canvas by the offset so no shifted crop overflows boundary.
    # Both move_masks and with_offset.to_dense() operate on identical space.
    new_h, new_w = image_height + dy, image_width + dx
    new_shape = (new_h, new_w)

    compact_offset_s = time_reps(
        lambda: compact_mask.with_offset(dx, dy, new_image_shape=new_shape)
    )
    if dense_skipped:
        return math.nan, compact_offset_s, None

    moved_dense = sv.move_masks(
        masks_dense, np.array([dx, dy]), resolution_wh=(new_w, new_h)
    )
    moved_compact = compact_mask.with_offset(
        dx, dy, new_image_shape=new_shape
    ).to_dense()
    offset_ok = bool(np.array_equal(moved_dense, moved_compact))
    dense_offset_s = time_reps(
        lambda: sv.move_masks(
            masks_dense, np.array([dx, dy]), resolution_wh=(new_w, new_h)
        )
    )
    return dense_offset_s, compact_offset_s, offset_ok


def stage_centroids(
    masks_dense: np.ndarray,
    compact_mask: CompactMask,
    dense_skipped: bool,
) -> tuple[float, float, bool | None]:
    """Time centroid: np.tensordot on full stack (dense) vs per-crop (compact)."""
    compact_centroids_s = time_reps(lambda: sv.calculate_masks_centroids(compact_mask))
    if dense_skipped:
        return math.nan, compact_centroids_s, None

    c_dense = sv.calculate_masks_centroids(masks_dense)
    c_compact = sv.calculate_masks_centroids(compact_mask)
    centroids_ok = bool(np.allclose(c_dense, c_compact, atol=1.0))  # 1-pixel tolerance
    dense_centroids_s = time_reps(lambda: sv.calculate_masks_centroids(masks_dense))
    return dense_centroids_s, compact_centroids_s, centroids_ok


# ══════════════════════════════════════════════════════════════════════════════
# Scenario runner — orchestrates stages
# ══════════════════════════════════════════════════════════════════════════════


def run_scenario(
    name: str,
    num_objects: int,
    image_height: int,
    image_width: int,
    fill_fraction: float = 0.10,
    num_vertices: int = 20,
) -> ScenarioResult:
    resolution = f"{image_width}x{image_height}"
    fill_name = f"{fill_fraction:.0%}"
    console.rule(
        f"[bold]{name}[/bold] | {num_objects} objects · {resolution} "
        f"· fill≈{fill_name} · polygon/{num_vertices} vertices"
    )

    xyxy, masks_dense, class_ids, compact_mask = stage_build(
        num_objects, image_height, image_width, fill_fraction, num_vertices
    )
    scene = make_scene(image_height, image_width)

    # ── memory ──────────────────────────────────────────────────────────────
    dense_bytes = dense_memory_bytes(masks_dense)
    dense_skipped = dense_bytes > DENSE_SKIP_GB * 1e9
    compact_theoretical = compact_memory_bytes_theoretical(compact_mask)

    # Only measure dense tracemalloc when it's safe to allocate the full array.
    dense_actual = (
        0
        if dense_skipped
        else dense_memory_bytes_actual(num_objects, image_height, image_width)
    )
    compact_actual = compact_memory_bytes_actual(
        masks_dense, xyxy, (image_height, image_width)
    )

    encode_s = stage_encode(masks_dense, xyxy, image_height, image_width)
    decode_s = stage_decode(compact_mask)

    theory_ratio = dense_bytes / max(compact_theoretical, 1)
    if dense_skipped:
        malloc_ratio_str = "[dim]—[/dim]"
        dense_actual_str = "[dim]skipped[/dim]"
    else:
        malloc_ratio = dense_actual / max(compact_actual, 1)
        malloc_ratio_str = _fmt_ratio(malloc_ratio)
        dense_actual_str = f"{dense_actual / 1e6:.1f} MB"
    console.print(
        f"\tmemory >>\n"
        f"\t\ttheory :: dense={dense_bytes / 1e6:.1f} MB "
        f"| compact={compact_theoretical / 1e3:.0f} KB "
        f"\t{_fmt_ratio(theory_ratio)}\n"
        f"\t\tmalloc :: dense={dense_actual_str} "
        f"| compact={compact_actual / 1e3:.0f} KB "
        f"\t{malloc_ratio_str}"
    )
    console.print(f"\t<create> encode (from_dense)\t={encode_s * 1e3:.3f} ms/mask")
    console.print(f"\t<export> decode (to_dense)\t={decode_s * 1e3:.3f} ms/mask")

    # ── skip flags ──────────────────────────────────────────────────────────
    iou_dense_skipped = dense_bytes > IOU_DENSE_SKIP_GB * 1e9
    if dense_skipped:
        console.print(
            f"\t[yellow]dense array is {dense_bytes / 1e9:.1f} GB "
            f"(>{DENSE_SKIP_GB:.0f} GB threshold) — skipping dense timing"
            f"[/yellow]"
        )
    elif iou_dense_skipped:
        console.print(
            f"\t[yellow]dense IoU skipped (>{IOU_DENSE_SKIP_GB:.0f}GB thr.)[/yellow]"
        )

    confidence = (
        np.random.default_rng(1).uniform(0.3, 0.99, num_objects).astype(np.float32)
    )
    det_compact = sv.Detections(xyxy=xyxy, mask=compact_mask, class_id=class_ids)

    if dense_skipped:
        dense_area_s = dense_filter_s = dense_annot_s = math.nan
        compact_area_s = _time_compact_area(det_compact)
        compact_filter_s = _time_compact_filter(det_compact)
        compact_annot_s = _time_compact_annotate(scene, det_compact)
        pixel_perfect = areas_match = roundtrip_ok = None
        det_dense = None
    else:
        det_dense = sv.Detections(xyxy=xyxy, mask=masks_dense, class_id=class_ids)
        dense_area_s, compact_area_s = stage_area(det_dense, det_compact)
        dense_filter_s, compact_filter_s = stage_filter(det_dense, det_compact)
        dense_annot_s, compact_annot_s = stage_annotate(scene, det_dense, det_compact)
        pixel_perfect, areas_match, roundtrip_ok = stage_correctness(
            scene, masks_dense, compact_mask, det_dense, det_compact
        )

    dense_iou_s, compact_iou_s, iou_ok = stage_iou(
        masks_dense, compact_mask, iou_dense_skipped
    )
    dense_nms_s, compact_nms_s, nms_ok, nms_diff = stage_nms(
        xyxy,
        confidence,
        class_ids,
        masks_dense,
        compact_mask,
        dense_skipped,
        iou_dense_skipped,
    )
    dense_merge_s, compact_merge_s, merge_ok = stage_merge(
        det_dense, det_compact, dense_skipped
    )
    dense_offset_s, compact_offset_s, offset_ok = stage_offset(
        masks_dense, compact_mask, image_height, image_width, dense_skipped
    )
    dense_centroids_s, compact_centroids_s, centroids_ok = stage_centroids(
        masks_dense, compact_mask, dense_skipped
    )

    def _timing_line(label: str, dense_s: float, compact_s: float) -> str:
        compact_ms = f"{compact_s * 1e3:.2f} ms"
        if math.isnan(dense_s):
            return (
                f"\t{label}\t -> dense=[dim]—[/dim]"
                f"\t\t | compact={compact_ms}\t | speedup=[dim]—[/dim]"
            )
        dense_ms = f"{dense_s * 1e3:.2f} ms"
        speedup = _fmt_ratio(dense_s / max(compact_s, 1e-9))
        return (
            f"\t{label}\t -> dense={dense_ms}\t | "
            f"compact={compact_ms}\t | speedup={speedup}"
        )

    console.print(_timing_line(".area    ", dense_area_s, compact_area_s))
    console.print(_timing_line("annotate ", dense_annot_s, compact_annot_s))
    console.print(_timing_line("centroids", dense_centroids_s, compact_centroids_s))
    console.print(_timing_line("filter   ", dense_filter_s, compact_filter_s))
    console.print(_timing_line("iou      ", dense_iou_s, compact_iou_s))
    console.print(_timing_line("merge    ", dense_merge_s, compact_merge_s))
    console.print(_timing_line("nms      ", dense_nms_s, compact_nms_s))
    console.print(_timing_line("offset   ", dense_offset_s, compact_offset_s))

    checks = {
        "pixel-perfect": pixel_perfect,
        "areas": areas_match,
        "roundtrip": roundtrip_ok,
        "iou": iou_ok,
        "nms": nms_ok,
        "merge": merge_ok,
        "offset": offset_ok,
        "centroids": centroids_ok,
    }
    parts = []
    for k, v in checks.items():
        if k == "nms" and v is False:
            parts.append(f"nms=[red]✗({nms_diff})[/red]")
        else:
            parts.append(
                f"{k}="
                + (
                    "[dim]—[/dim]"
                    if v is None
                    else "[green]✓[/green]"
                    if v
                    else "[red]✗[/red]"
                )
            )
    all_checked = [v for v in checks.values() if v is not None]
    overall = (
        "[green]✓ all correct[/green]"
        if all_checked and all(all_checked)
        else "[red]✗ MISMATCH[/red]"
        if any(v is False for v in checks.values())
        else "[dim]—[/dim]"
    )
    console.print("  correctness >> " + " | ".join(parts) + f" | {overall}")

    return ScenarioResult(
        name=name,
        resolution=resolution,
        num_objects=num_objects,
        fill_name=fill_name,
        num_vertices=num_vertices,
        dense_bytes=dense_bytes,
        compact_bytes_theoretical=compact_theoretical,
        dense_bytes_actual=dense_actual,
        compact_bytes_actual=compact_actual,
        encode_s=encode_s,
        decode_s=decode_s,
        dense_area_s=dense_area_s,
        compact_area_s=compact_area_s,
        dense_filter_s=dense_filter_s,
        compact_filter_s=compact_filter_s,
        dense_annot_s=dense_annot_s,
        compact_annot_s=compact_annot_s,
        dense_iou_s=dense_iou_s,
        compact_iou_s=compact_iou_s,
        dense_nms_s=dense_nms_s,
        compact_nms_s=compact_nms_s,
        dense_merge_s=dense_merge_s,
        compact_merge_s=compact_merge_s,
        dense_offset_s=dense_offset_s,
        compact_offset_s=compact_offset_s,
        dense_centroids_s=dense_centroids_s,
        compact_centroids_s=compact_centroids_s,
        pixel_perfect=pixel_perfect,
        areas_match=areas_match,
        roundtrip_ok=roundtrip_ok,
        iou_ok=iou_ok,
        nms_ok=nms_ok,
        nms_mismatch_count=nms_diff,
        merge_ok=merge_ok,
        offset_ok=offset_ok,
        centroids_ok=centroids_ok,
        dense_skipped=dense_skipped,
        iou_dense_skipped=iou_dense_skipped,
    )


def _time_compact_area(det_compact: sv.Detections) -> float:
    """Time .area on the compact detections (used when dense timing is skipped)."""
    return time_reps(lambda: det_compact.area)


def _time_compact_filter(det_compact: sv.Detections) -> float:
    """Time boolean-index filtering on the compact detections (dense-skip path)."""
    keep = np.arange(len(det_compact)) % 2 == 0
    return time_reps(lambda: det_compact[keep])


def _time_compact_annotate(scene: np.ndarray, det_compact: sv.Detections) -> float:
    """Time MaskAnnotator on the compact detections (dense-skip path)."""
    annotator = sv.MaskAnnotator(opacity=0.5)
    return time_reps(lambda: annotator.annotate(scene.copy(), det_compact))


# ══════════════════════════════════════════════════════════════════════════════
# Rich summary table
# ══════════════════════════════════════════════════════════════════════════════

_OPS = ("area", "filter", "annot", "iou", "nms", "merge", "offset", "centroids")


def _build_summary_df(results: list[ScenarioResult]) -> pd.DataFrame:
    """Compute derived summary columns from scenario results.

    Returns a DataFrame with all ScenarioResult fields plus derived columns
    (ratios, speedups, ok) as raw floats.  Consumers apply their own formatting.
    """
    df = pd.DataFrame([dataclasses.asdict(r) for r in results])
    df["ratio_theory"] = df["dense_bytes"] / df["compact_bytes_theoretical"].clip(
        lower=1
    )
    df["ratio_malloc"] = df["dense_bytes_actual"] / df["compact_bytes_actual"].clip(
        lower=1
    )
    # dense_bytes_actual == 0 (not measured) when dense_skipped — clear those cells
    df.loc[df["dense_skipped"], "ratio_malloc"] = None
    for op in _OPS:
        df[f"{op}_speedup"] = df[f"dense_{op}_s"] / df[f"compact_{op}_s"].clip(
            lower=1e-9
        )

    check_cols = [
        "pixel_perfect",
        "areas_match",
        "roundtrip_ok",
        "iou_ok",
        "nms_ok",
        "merge_ok",
        "offset_ok",
        "centroids_ok",
    ]
    df["ok"] = df.apply(
        lambda row: (
            False
            if any(row[c] is False for c in check_cols)
            else True
            if any(row[c] is True for c in check_cols)
            else None
        ),
        axis=1,
    )
    return df


def _fmt_ratio(ratio: float) -> str:
    """Format a speedup/compression ratio with colour coding.

    ≥10 → green (large win), 1-10 → yellow (modest win), <1 → red (regression).
    Integer for ≥10, two decimals otherwise.
    """
    fmt = f"{ratio:.0f}x" if ratio >= 10 else f"{ratio:.2f}x"
    if ratio >= 10:
        return f"[green]{fmt}[/green]"
    elif ratio >= 1:
        return f"[yellow]{fmt}[/yellow]"
    else:
        return f"[red]{fmt}[/red]"


def _fmt_speedup(dense_s: float, compact_s: float) -> str:
    if math.isnan(dense_s):
        # Dense was skipped — show compact absolute time so the column isn't empty.
        return f"[dim]{compact_s * 1e3:.1f} ms[/dim]"
    return _fmt_ratio(dense_s / max(compact_s, 1e-9))


def print_summary(results: list[ScenarioResult]) -> None:
    table = Table(
        title="CompactMask — benchmark summary",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
        min_width=console.width,
    )
    table.add_column("Scenario", style="bold", min_width=22)
    table.add_column("Objects", justify="right", min_width=7)
    table.add_column("Resolution", min_width=12, no_wrap=True)
    table.add_column("Fill", justify="right", min_width=5, no_wrap=True)
    table.add_column("Vertices", justify="right", min_width=8, no_wrap=True)
    table.add_column("Dense\ntheory", justify="right", min_width=10)
    table.add_column("Compact\ntheory", justify="right", style="green", min_width=9)
    table.add_column("Ratio\ntheory", justify="right", min_width=7)
    table.add_column("Dense\nmalloc", justify="right", style="cyan", min_width=9)
    table.add_column("Compact\nmalloc", justify="right", style="cyan", min_width=9)
    table.add_column("Ratio\nmalloc", justify="right", min_width=7)
    table.add_column("Encode\n(ms/mask)", justify="right", style="yellow", min_width=7)
    table.add_column("Decode\n(ms/mask)", justify="right", style="yellow", min_width=7)
    table.add_column("Area\natt.", justify="right", min_width=6)
    table.add_column("Filter\nop.", justify="right", min_width=6)
    table.add_column("Annot\nop.", justify="right", min_width=6)
    table.add_column("IoU\nop.", justify="right", min_width=6)
    table.add_column("NMS\nop.", justify="right", min_width=6)
    table.add_column("Merge\nop.", justify="right", min_width=6)
    table.add_column("Offset\nop.", justify="right", min_width=6)
    table.add_column("Centr\nop.", justify="right", min_width=6)
    table.add_column("OK?", justify="center", min_width=4)

    for _, row in _build_summary_df(results).iterrows():
        ok = row["ok"]
        ok_cell = (
            "[red]✗[/red]"
            if ok is False
            else "[green]✓[/green]"
            if ok is True
            else "[dim]—[/dim]"
        )
        dense_malloc_cell = (
            "[dim]—[/dim]"
            if row["dense_skipped"]
            else f"{row['dense_bytes_actual'] / 1e6:.1f} MB"
        )
        malloc_ratio_cell = (
            "[dim]—[/dim]" if row["dense_skipped"] else _fmt_ratio(row["ratio_malloc"])
        )
        table.add_row(
            row["name"],
            str(row["num_objects"]),
            row["resolution"],
            row["fill_name"],
            str(row["num_vertices"]),
            f"{row['dense_bytes'] / 1e6:.1f} MB",
            f"{row['compact_bytes_theoretical'] / 1e3:.0f} KB",
            _fmt_ratio(row["ratio_theory"]),
            dense_malloc_cell,
            f"{row['compact_bytes_actual'] / 1e3:.0f} KB",
            malloc_ratio_cell,
            f"{row['encode_s'] * 1e3:.1f}",
            f"{row['decode_s'] * 1e3:.1f}",
            _fmt_speedup(row["dense_area_s"], row["compact_area_s"]),
            _fmt_speedup(row["dense_filter_s"], row["compact_filter_s"]),
            _fmt_speedup(row["dense_annot_s"], row["compact_annot_s"]),
            _fmt_speedup(row["dense_iou_s"], row["compact_iou_s"]),
            _fmt_speedup(row["dense_nms_s"], row["compact_nms_s"]),
            _fmt_speedup(row["dense_merge_s"], row["compact_merge_s"]),
            _fmt_speedup(row["dense_offset_s"], row["compact_offset_s"]),
            _fmt_speedup(row["dense_centroids_s"], row["compact_centroids_s"]),
            ok_cell,
        )

    console.print(table)
    console.print(
        "[dim]"
        + "  ·  ".join(
            [
                "Vertices — polygon vertex count "
                "(complexity proxy: more = jaggier boundary)",
                "Dense theory — NxHxW bytes (raw numpy buffer)",
                "Compact theory — sum of internal numpy buffer sizes",
                "Ratio (theory) — dense / compact theoretical ratio",
                "Dense malloc — tracemalloc peak during np.zeros allocation",
                "Compact malloc — tracemalloc peak during .from_dense()",
                "Ratio (malloc) — dense / compact tracemalloc peak ratio",
                "Encode ms/mask — from_dense() / N (dense→compact overhead per mask)",
                "Decode ms/mask — to_dense() / N (compact→dense overhead per mask)",
                "Area x — .area speedup (RLE sum, no materialisation)",
                "Filter x — boolean-index speedup",
                "Annot x — MaskAnnotator speedup (crop-paint vs full-frame alloc)",
                f"IoU x — pairwise self-IoU speedup "
                f"(dense skipped >{IOU_DENSE_SKIP_GB:.0f} GB)",
                "NMS x — mask_non_max_suppression speedup",
                "Merge x — Detections.merge speedup",
                "Offset x — move_masks vs with_offset speedup",
                "Centroids x — calculate_masks_centroids speedup",
                "dim ms — dense skipped, compact absolute time shown",
            ]
        )
        + "[/dim]"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Results persistence
# ══════════════════════════════════════════════════════════════════════════════


def _append_result(result: ScenarioResult, path: Path) -> None:
    """Append one scenario result as a JSON line to *path*.

    ``math.nan`` (used for skipped dense timings) is serialised as ``null``
    so the file is valid JSON-Lines and can be read back with any JSON parser.
    """
    row = {
        k: (None if isinstance(v, float) and math.isnan(v) else v)
        for k, v in dataclasses.asdict(result).items()
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def save_results_csv(results: list[ScenarioResult], path: Path) -> None:
    """Write the summary table to *path* as a CSV file.

    Each row mirrors the Rich summary table: scenario metadata, memory ratios,
    encode/decode overhead, and per-operation speedups. Columns whose dense
    timing was skipped are written as empty cells.
    """
    df = _build_summary_df(results)
    pd.DataFrame(
        {
            "scenario": df["name"],
            "objects": df["num_objects"],
            "resolution": df["resolution"],
            "fill": df["fill_name"],
            "vertices": df["num_vertices"],
            "dense_theory_mb": (df["dense_bytes"] / 1e6).round(1),
            "compact_theory_kb": (df["compact_bytes_theoretical"] / 1e3).round(1),
            "ratio_theory": df["ratio_theory"].round(0),
            "dense_malloc_mb": (df["dense_bytes_actual"] / 1e6)
            .where(~df["dense_skipped"])
            .round(1),
            "compact_malloc_kb": (df["compact_bytes_actual"] / 1e3).round(1),
            "ratio_malloc": df["ratio_malloc"].round(0),
            "encode_ms_per_mask": (df["encode_s"] * 1e3).round(4),
            "decode_ms_per_mask": (df["decode_s"] * 1e3).round(4),
            **{f"{op}_speedup": df[f"{op}_speedup"].round(2) for op in _OPS},
            "ok": df["ok"],
        }
    ).to_csv(path, index=False)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    # ── parameter matrix ──────────────────────────────────────────────────────
    # (tier_label, (image_width, image_height), num_objects)
    TIERS: list[tuple[str, tuple[int, int], int]] = [
        ("FHD", (1920, 1080), 100),  # full comparison  (0.21 GB < 1 GB IoU thr.)
        ("FHD", (1920, 1080), 200),  # full comparison  (0.41 GB < 1 GB IoU thr.)
        ("FHD", (1920, 1080), 400),  # full comparison  (0.83 GB < 1 GB IoU thr.)
        ("4K", (3840, 2160), 100),  # full comparison  (0.83 GB < 1 GB IoU thr.)
        ("4K", (3840, 2160), 200),  # dense excl. IoU/NMS  (1.66 GB > 1 GB thr.)
        ("SAT", (8192, 8192), 200),  # dense excl. IoU/NMS  (13.4 GB > 1 GB thr.)
    ]
    FILL_FRACTIONS = [0.05, 0.20, 0.50]  # sparse / moderate / SAM-everything
    VERTEX_COUNTS = [8, 128, 600]  # low / realistic / YOLOv8-seg default

    scenarios = [
        {
            "name": f"{tier}-{num_objects}-{fill_fraction:.0%}-v{num_vertices}",
            "num_objects": num_objects,
            "image_height": img_h,
            "image_width": img_w,
            "fill_fraction": fill_fraction,
            "num_vertices": num_vertices,
        }
        for tier, (img_w, img_h), num_objects in TIERS
        for fill_fraction in FILL_FRACTIONS
        for num_vertices in VERTEX_COUNTS
    ]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_path = Path(__file__).parent / f"results_{timestamp}.jsonl"

    console.print(
        f"[bold]supervision[/bold]"
        f" {sv.__version__}  ·  numpy {np.__version__}  ·  {len(scenarios)} scenarios"
        f"  ·  saving to [dim]{results_path.name}[/dim]"
    )

    results = []
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("benchmarking…", total=len(scenarios))
        for params in scenarios:
            progress.update(task, description=f"[bold]{params['name']}[/bold]")
            result = run_scenario(**params)
            results.append(result)
            _append_result(result, results_path)
            gc.collect()  # flush scenario temporaries before next run
            progress.advance(task)

    print_summary(results)

    csv_path = results_path.with_suffix(".csv")
    save_results_csv(results, csv_path)
    console.print(f"[dim]results saved → {results_path.name}  ·  {csv_path.name}[/dim]")


if __name__ == "__main__":
    main()
