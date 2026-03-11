"""CompactMask demo & benchmark.

Demonstrates that ``CompactMask`` is a drop-in replacement for dense
``(N, H, W)`` bool arrays in ``supervision.Detections``, while using
significantly less memory and enabling faster annotation.

Run with:
    uv run python examples/compact_mask/benchmark.py

No GPU or real model is required — everything is synthesized with NumPy.
"""

from __future__ import annotations

import functools
import math
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table

import supervision as sv
from supervision.detection.compact_mask import CompactMask

console = Console(width=140, force_terminal=True)

REPS = 5
# Dense timing is skipped when the dense (N,H,W) array would exceed this
# threshold — avoids OOM / swap thrashing on large satellite scenarios while
# still reporting the theoretical memory footprint.
DENSE_SKIP_GB = 16.0


# ══════════════════════════════════════════════════════════════════════════════
# Result container
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ScenarioResult:
    name: str
    resolution: str  # e.g. "1920x1080"
    num_objects: int
    fill_name: str  # e.g. "5%"
    # memory (theoretical: raw numpy nbytes)
    dense_bytes: int
    compact_bytes_theoretical: int
    # memory (actual: tracemalloc peak for CompactMask object itself)
    compact_bytes_actual: int
    # timing (nan when dense_skipped=True)
    dense_area_s: float
    compact_area_s: float
    dense_filter_s: float
    compact_filter_s: float
    dense_annotate_s: float
    compact_annotate_s: float
    # correctness
    pixel_perfect: bool
    areas_match: bool
    roundtrip_ok: bool
    # whether dense timing was skipped due to DENSE_SKIP_GB threshold
    dense_skipped: bool = field(default=False)


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic data helpers
# ══════════════════════════════════════════════════════════════════════════════


def make_scene(image_height: int, image_width: int) -> np.ndarray:
    """Random BGR image."""
    return np.random.default_rng(42).integers(
        0, 255, (image_height, image_width, 3), dtype=np.uint8
    )


@functools.cache
def make_detections(
    num_objects: int,
    image_height: int,
    image_width: int,
    fill_fraction: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(xyxy, masks_dense, class_ids)`` with ellipse-shaped masks.

    Results are cached so the same parameter combination is only synthesized
    once across the full benchmark run.
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
        ellipse_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        cv2.ellipse(
            ellipse_mask, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 1, -1
        )
        masks[index] = ellipse_mask.astype(bool)
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
        + sum(rle.nbytes for rle in compact_mask._rles)
    )


def measure_peak_bytes(func: Callable[[], object]) -> int:
    """Wrapper that runs *func* under tracemalloc and returns the peak allocation.

    tracemalloc captures every Python-level allocation — numpy buffers, list
    nodes, object headers — giving the true heap cost of anything *func* builds.
    The return value of *func* is discarded so the object does not stay alive.
    """
    tracemalloc.start()
    tracemalloc.clear_traces()
    func()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return int(peak)


def compact_memory_bytes_actual(
    masks_dense: np.ndarray,
    xyxy: np.ndarray,
    image_shape: tuple[int, int],
) -> int:
    """Actual compact footprint: peak bytes during CompactMask.from_dense()."""
    return measure_peak_bytes(
        lambda: CompactMask.from_dense(masks_dense, xyxy, image_shape=image_shape)
    )


def time_reps(func: Callable[[], object], reps: int = REPS) -> float:
    """Run *func* *reps* times and return the mean wall-clock seconds per call."""
    t0 = time.perf_counter()
    for _ in range(reps):
        func()
    return (time.perf_counter() - t0) / reps


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark stages
# ══════════════════════════════════════════════════════════════════════════════


def stage_build(
    num_objects: int, image_height: int, image_width: int, fill_fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, CompactMask]:
    """Synthesize dense masks and build the CompactMask from them."""
    xyxy, masks_dense, class_ids = make_detections(
        num_objects, image_height, image_width, fill_fraction
    )
    compact_mask = CompactMask.from_dense(
        masks_dense, xyxy, image_shape=(image_height, image_width)
    )
    return xyxy, masks_dense, class_ids, compact_mask


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


# ══════════════════════════════════════════════════════════════════════════════
# Scenario runner — orchestrates stages
# ══════════════════════════════════════════════════════════════════════════════


def run_scenario(
    name: str,
    num_objects: int,
    image_height: int,
    image_width: int,
    fill_fraction: float = 0.10,
) -> ScenarioResult:
    resolution = f"{image_width}x{image_height}"
    fill_name = f"{fill_fraction:.0%}"
    console.rule(
        f"[bold]{name}[/bold]  {num_objects} objects · {resolution} · fill≈{fill_name}"
    )

    with console.status("  building masks…"):
        xyxy, masks_dense, class_ids, compact_mask = stage_build(
            num_objects, image_height, image_width, fill_fraction
        )
        scene = make_scene(image_height, image_width)

    # ── memory ──────────────────────────────────────────────────────────────
    dense_bytes = dense_memory_bytes(masks_dense)
    compact_theoretical = compact_memory_bytes_theoretical(compact_mask)

    with console.status("  measuring actual CompactMask allocation…"):
        compact_actual = compact_memory_bytes_actual(
            masks_dense, xyxy, (image_height, image_width)
        )

    mem_ratio = dense_bytes / max(compact_theoretical, 1)
    console.print(
        f"  memory  dense={dense_bytes / 1e6:.1f} MB  "
        f"compact theoretical={compact_theoretical / 1e3:.0f} KB  "
        f"compact actual (tracemalloc)={compact_actual / 1e3:.0f} KB  "
        f"[green]ratio {mem_ratio:.0f}x[/green]"
    )

    # ── decide whether to skip dense timing ─────────────────────────────────
    dense_skipped = dense_bytes > DENSE_SKIP_GB * 1e9
    if dense_skipped:
        console.print(
            f"  [yellow]dense array is {dense_bytes / 1e9:.1f} GB "
            f"(>{DENSE_SKIP_GB:.0f} GB threshold) — skipping dense timing[/yellow]"
        )

    det_compact = sv.Detections(xyxy=xyxy, mask=compact_mask, class_id=class_ids)

    if dense_skipped:
        det_dense = None
        dense_area_s = math.nan
        compact_area_s = _time_compact_area(det_compact)
        dense_filter_s = math.nan
        compact_filter_s = _time_compact_filter(det_compact)
        dense_annotate_s = math.nan
        compact_annotate_s = _time_compact_annotate(scene, det_compact)
        pixel_perfect = None  # correctness proven on smaller scenarios
        areas_match = None
        roundtrip_ok = None
    else:
        det_dense = sv.Detections(xyxy=xyxy, mask=masks_dense, class_id=class_ids)
        dense_area_s, compact_area_s = stage_area(det_dense, det_compact)
        dense_filter_s, compact_filter_s = stage_filter(det_dense, det_compact)
        with console.status("  annotating…"):
            dense_annotate_s, compact_annotate_s = stage_annotate(
                scene, det_dense, det_compact
            )
        with console.status("  checking correctness…"):
            pixel_perfect, areas_match, roundtrip_ok = stage_correctness(
                scene, masks_dense, compact_mask, det_dense, det_compact
            )

    def _timing_line(label: str, dense_s: float, compact_s: float) -> str:
        compact_ms = f"{compact_s * 1e3:.2f} ms"
        if math.isnan(dense_s):
            return f"  {label} - compact={compact_ms}"
        dense_ms = f"{dense_s * 1e3:.2f} ms"
        speedup = _fmt_ratio(dense_s / max(compact_s, 1e-9))
        return (
            f"  {label}\t "
            f"-> dense={dense_ms}\t | compact={compact_ms}\t | speedup={speedup}"
        )

    console.print(_timing_line(".area  ", dense_area_s, compact_area_s))
    console.print(_timing_line("filter ", dense_filter_s, compact_filter_s))
    console.print(_timing_line("annotate", dense_annotate_s, compact_annotate_s))
    if not dense_skipped:
        all_correct = pixel_perfect and areas_match and roundtrip_ok
        status = (
            "[green]✓ all correct[/green]" if all_correct else "[red]✗ MISMATCH[/red]"
        )
        console.print(
            f"  correctness ->  pixel-perfect={pixel_perfect} | "
            f"areas={areas_match} | roundtrip={roundtrip_ok} | {status}"
        )

    return ScenarioResult(
        name=name,
        resolution=resolution,
        num_objects=num_objects,
        fill_name=fill_name,
        dense_bytes=dense_bytes,
        compact_bytes_theoretical=compact_theoretical,
        compact_bytes_actual=compact_actual,
        dense_area_s=dense_area_s,
        compact_area_s=compact_area_s,
        dense_filter_s=dense_filter_s,
        compact_filter_s=compact_filter_s,
        dense_annotate_s=dense_annotate_s,
        compact_annotate_s=compact_annotate_s,
        pixel_perfect=pixel_perfect,
        areas_match=areas_match,
        roundtrip_ok=roundtrip_ok,
        dense_skipped=dense_skipped,
    )


def _time_compact_area(det_compact: sv.Detections) -> float:
    return time_reps(lambda: det_compact.area)


def _time_compact_filter(det_compact: sv.Detections) -> float:
    keep = np.arange(len(det_compact)) % 2 == 0
    return time_reps(lambda: det_compact[keep])


def _time_compact_annotate(scene: np.ndarray, det_compact: sv.Detections) -> float:
    annotator = sv.MaskAnnotator(opacity=0.5)
    return time_reps(lambda: annotator.annotate(scene.copy(), det_compact))


# ══════════════════════════════════════════════════════════════════════════════
# Rich summary table
# ══════════════════════════════════════════════════════════════════════════════


def _fmt_ratio(ratio: float) -> str:
    """Format a speedup ratio — one decimal place so 0.57x is not rounded to 1x."""
    return f"{ratio:.1f}x"


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
        min_width=100,
    )
    table.add_column("Scenario", style="bold", min_width=13)
    table.add_column("Objects", justify="right", min_width=7)
    table.add_column("Resolution", min_width=12, no_wrap=True)
    table.add_column("Fill", justify="right", min_width=5, no_wrap=True)
    table.add_column("Dense mem", justify="right", min_width=10)
    table.add_column("Compact\ntheory", justify="right", style="green", min_width=9)
    table.add_column("Compact\nactual", justify="right", style="cyan", min_width=9)
    table.add_column("Mem\n(x)", justify="right", style="green", min_width=7)
    table.add_column("Area\n(x)", justify="right", style="green", min_width=7)
    table.add_column("Filter\n(x)", justify="right", style="green", min_width=9)
    table.add_column("Annot\n(x)", justify="right", style="green", min_width=8)
    table.add_column("OK?", justify="center", min_width=4)

    for result in results:
        mem_ratio = result.dense_bytes / max(result.compact_bytes_theoretical, 1)
        all_correct = (
            result.pixel_perfect and result.areas_match and result.roundtrip_ok
        )
        ok_cell = (
            "[dim]—[/dim]"
            if result.dense_skipped
            else ("[green]✓[/green]" if all_correct else "[red]✗[/red]")
        )
        table.add_row(
            result.name,
            str(result.num_objects),
            result.resolution,
            result.fill_name,
            f"{result.dense_bytes / 1e6:.1f} MB",
            f"{result.compact_bytes_theoretical / 1e3:.0f} KB",
            f"{result.compact_bytes_actual / 1e3:.0f} KB",
            f"{mem_ratio:.0f}x",
            _fmt_speedup(result.dense_area_s, result.compact_area_s),
            _fmt_speedup(result.dense_filter_s, result.compact_filter_s),
            _fmt_speedup(result.dense_annotate_s, result.compact_annotate_s),
            ok_cell,
        )

    console.print()
    console.print(table)
    console.print(
        "  ·  ".join(
            [
                "[dim]",
                "Compact theor. — sum of internal numpy buffer sizes",
                "Compact actual — tracemalloc peak during CompactMask.from_dense()"
                " (w/ Python overhead)",
                "Mem x — dense / compact theoretical ratio",
                "Area x — .area speedup (RLE sum, no materialisation)",
                "Filter x — boolean-index speedup",
                "Annot x — MaskAnnotator speedup (crop-paint vs full-frame alloc)",
                f"italic ms — dense skipped (array > {DENSE_SKIP_GB:.0f} GB),"
                f" compact absolute time shown[/dim]",
            ]
        )
    )


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    console.print(
        f"[bold]supervision[/bold] {sv.__version__}  ·  numpy {np.__version__}"
    )

    results = [
        # Full HD — typical video frame
        run_scenario(
            "FHD-100-5%",
            num_objects=100,
            image_height=1080,
            image_width=1920,
            fill_fraction=0.05,
        ),
        run_scenario(
            "FHD-100-10%",
            num_objects=100,
            image_height=1080,
            image_width=1920,
            fill_fraction=0.10,
        ),
        run_scenario(
            "FHD-100-20%",
            num_objects=100,
            image_height=1080,
            image_width=1920,
            fill_fraction=0.20,
        ),
        # 4K — drone / cinema
        run_scenario(
            "4K-500-5%",
            num_objects=500,
            image_height=2160,
            image_width=3840,
            fill_fraction=0.05,
        ),
        run_scenario(
            "4K-500-10%",
            num_objects=500,
            image_height=2160,
            image_width=3840,
            fill_fraction=0.10,
        ),
        run_scenario(
            "4K-500-20%",
            num_objects=500,
            image_height=2160,
            image_width=3840,
            fill_fraction=0.20,
        ),
        run_scenario(
            "4K-1000-5%",
            num_objects=1000,
            image_height=2160,
            image_width=3840,
            fill_fraction=0.05,
        ),
        run_scenario(
            "4K-1000-10%",
            num_objects=1000,
            image_height=2160,
            image_width=3840,
            fill_fraction=0.10,
        ),
        run_scenario(
            "4K-1000-20%",
            num_objects=1000,
            image_height=2160,
            image_width=3840,
            fill_fraction=0.20,
        ),
        # 8192x8192 — common satellite / GeoTIFF benchmark tile (Sentinel-2 class)
        run_scenario(
            "SAT-200-5%",
            num_objects=200,
            image_height=8192,
            image_width=8192,
            fill_fraction=0.05,
        ),
        run_scenario(
            "SAT-200-10%",
            num_objects=200,
            image_height=8192,
            image_width=8192,
            fill_fraction=0.10,
        ),
        run_scenario(
            "SAT-200-20%",
            num_objects=200,
            image_height=8192,
            image_width=8192,
            fill_fraction=0.20,
        ),
    ]

    print_summary(results)


if __name__ == "__main__":
    main()
