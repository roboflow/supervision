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

By default, scenarios are derived from segmentation artifact images in
``examples/compact_mask/outputs``. Use ``--synthetic`` to run generated polygon
masks instead.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
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

from supervision.detection.compact_mask import CompactMask
from supervision.detection.utils.converters import mask_to_xyxy

console = Console(width=120, force_terminal=True)

REPETITIONS = 20
WARMUP = 3
DEFAULT_SOURCE_DIR = Path("examples/compact_mask/outputs")
MAX_IMAGE_DIMENSION = 960
MAX_SEGMENTS_PER_IMAGE = 120
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


@dataclass(frozen=True, slots=True)
class Scenario:
    """Synthetic painting benchmark scenario."""

    name: str
    image_shape: tuple[int, int]
    num_objects: int
    fill_fraction: float
    vertices: int


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
    mask_area_ratio: float
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


def make_scene(image_shape: tuple[int, int]) -> np.ndarray:
    """Create a deterministic BGR scene."""
    image_h, image_w = image_shape
    rng = np.random.default_rng(17)
    return rng.integers(0, 255, (image_h, image_w, 3), dtype=np.uint8)


def resize_for_benchmark(image: np.ndarray, max_dimension: int) -> np.ndarray:
    """Resize ``image`` so the largest side is at most ``max_dimension``."""
    image_h, image_w = image.shape[:2]
    scale = min(1.0, max_dimension / max(image_h, image_w))
    if scale == 1.0:
        return image
    size_wh = (max(1, int(image_w * scale)), max(1, int(image_h * scale)))
    return cv2.resize(image, size_wh, interpolation=cv2.INTER_AREA)


def make_masks(
    image_shape: tuple[int, int],
    num_objects: int,
    fill_fraction: float,
    vertices: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create dense masks and tight ``xyxy`` boxes for one scenario."""
    image_h, image_w = image_shape
    rng = np.random.default_rng(seed)
    masks = np.zeros((num_objects, image_h, image_w), dtype=bool)
    target_area = image_h * image_w * fill_fraction / max(1, num_objects)
    radius = max(4, int((target_area / np.pi) ** 0.5))

    for index in range(num_objects):
        center_x = int(rng.integers(radius + 1, max(radius + 2, image_w - radius - 1)))
        center_y = int(rng.integers(radius + 1, max(radius + 2, image_h - radius - 1)))
        angles = np.linspace(0, 2 * np.pi, vertices, endpoint=False)
        angles += rng.uniform(-np.pi / vertices, np.pi / vertices, size=vertices)
        radii = radius * rng.uniform(0.45, 1.25, size=vertices)
        points = np.column_stack(
            [
                np.clip(center_x + radii * np.cos(angles), 0, image_w - 1),
                np.clip(center_y + radii * np.sin(angles), 0, image_h - 1),
            ]
        ).astype(np.int32)
        canvas = np.zeros((image_h, image_w), dtype=np.uint8)
        cv2.fillPoly(canvas, [points.reshape(-1, 1, 2)], 1)
        masks[index] = canvas.astype(bool)

    return masks, mask_to_xyxy(masks).astype(np.float32)


def make_synthetic_input(scenario: Scenario) -> PaintInput:
    """Create one synthetic benchmark input."""
    scene = make_scene(scenario.image_shape)
    masks, xyxy = make_masks(
        image_shape=scenario.image_shape,
        num_objects=scenario.num_objects,
        fill_fraction=scenario.fill_fraction,
        vertices=scenario.vertices,
        seed=42,
    )
    return PaintInput(scenario.name, scene, masks, xyxy)


def masks_from_image_segments(
    image: np.ndarray,
    max_segments: int,
    clusters: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive deterministic semantic-style masks from image color regions."""
    image_h, image_w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    samples = lab.reshape((-1, 3)).astype(np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        20,
        1.0,
    )
    cv2.setRNGSeed(42)
    _, labels, _ = cv2.kmeans(
        samples,
        clusters,
        None,
        criteria,
        1,
        cv2.KMEANS_PP_CENTERS,
    )
    labels_2d = labels.reshape((image_h, image_w))
    kernel = np.ones((3, 3), dtype=np.uint8)
    min_area = max(32, int(image_h * image_w * 0.0005))

    components: list[tuple[int, np.ndarray]] = []
    for label in range(clusters):
        cluster_mask = (labels_2d == label).astype(np.uint8)
        cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_OPEN, kernel)
        cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, kernel)
        num_labels, component_labels, stats, _ = cv2.connectedComponentsWithStats(
            cluster_mask,
            connectivity=8,
        )
        for component_idx in range(1, num_labels):
            area = int(stats[component_idx, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            components.append((area, component_labels == component_idx))

    components.sort(key=lambda item: item[0], reverse=True)
    masks = [mask for _, mask in components[:max_segments]]
    if not masks:
        raise ValueError("No image segments found.")
    masks_arr = np.asarray(masks, dtype=bool)
    return masks_arr, mask_to_xyxy(masks_arr).astype(np.float32)


def image_paths(source_dir: Path, max_images: int | None) -> list[Path]:
    """Return image paths to use for image-backed benchmark cases."""
    paths = sorted(
        path
        for path in source_dir.glob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if max_images is not None:
        paths = paths[:max_images]
    return paths


def make_image_input(path: Path, max_segments: int, max_dimension: int) -> PaintInput:
    """Create one image-backed benchmark input from a segmentation artifact."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    scene = resize_for_benchmark(image, max_dimension)
    masks, xyxy = masks_from_image_segments(scene, max_segments=max_segments)
    name = path.stem.removesuffix("_segmentations")
    return PaintInput(name, scene, masks, xyxy)


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
    """Paint CompactMask directly from RLE true spans."""
    for detection_idx in order:
        x1 = int(compact_mask.offsets[detection_idx, 0])
        y1 = int(compact_mask.offsets[detection_idx, 1])
        color = color_for_index(int(detection_idx))
        for crop_x, span_y1, span_y2 in compact_mask._iter_true_spans(
            int(detection_idx)
        ):
            image_x = x1 + crop_x
            image_y1 = y1 + span_y1
            image_y2 = y1 + span_y2
            canvas[image_y1:image_y2, image_x] = color


def paint_full_dense(
    scene: np.ndarray,
    masks: np.ndarray,
    order: np.ndarray,
) -> np.ndarray:
    """Paint full-frame dense masks by boolean indexing."""
    canvas = scene.copy()
    paint_full_dense_into(canvas, masks, order)
    return canvas


def paint_crop_dense(
    scene: np.ndarray,
    compact_mask: CompactMask,
    order: np.ndarray,
) -> np.ndarray:
    """Paint CompactMask by decoding each crop to a dense boolean array."""
    canvas = scene.copy()
    paint_crop_dense_into(canvas, compact_mask, order)
    return canvas


def paint_direct_rle(
    scene: np.ndarray,
    compact_mask: CompactMask,
    order: np.ndarray,
) -> np.ndarray:
    """Paint CompactMask directly from RLE true spans."""
    canvas = scene.copy()
    paint_direct_rle_into(canvas, compact_mask, order)
    return canvas


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

    def full_dense() -> np.ndarray:
        return paint_full_dense(scene, masks, order)

    def crop_dense() -> np.ndarray:
        return paint_crop_dense(scene, compact_mask, order)

    def direct_rle() -> np.ndarray:
        return paint_direct_rle(scene, compact_mask, order)

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

    dense_result = full_dense()
    crop_result = crop_dense()
    rle_result = direct_rle()

    image_h, image_w = scene.shape[:2]
    return PaintBenchmarkResult(
        scenario=paint_input.name,
        resolution=f"{image_w}x{image_h}",
        objects=len(masks),
        mask_area_ratio=float(masks.sum() / max(1, masks.shape[1] * masks.shape[2])),
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


def run_scenario(
    scenario: Scenario,
    repetitions: int,
    warmup: int,
) -> PaintBenchmarkResult:
    """Run all painting strategies for one synthetic scenario."""
    return run_input(make_synthetic_input(scenario), repetitions, warmup)


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


def print_summary(
    results: list[PaintBenchmarkResult],
    reps: int,
    warmup: int,
) -> None:
    """Print a Rich summary table matching the compact mask benchmark style."""
    table = Table(
        title="CompactMask mask painting",
        box=box.ROUNDED,
        show_lines=False,
        header_style="bold cyan",
    )
    table.add_column("src", style="bold", no_wrap=True)
    table.add_column("res", no_wrap=True)
    table.add_column("seg", justify="right")
    table.add_column("area %", justify="right")
    table.add_column("full ms", justify="right")
    table.add_column("crop ms", justify="right")
    table.add_column("RLE ms", justify="right", style="green")
    table.add_column("RLE/full", justify="right")
    table.add_column("RLE/crop", justify="right")
    table.add_column("runs/spans", justify="right")
    table.add_column("scratch MB", justify="right", style="cyan")
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
            f"{result.mask_area_ratio * 100:.2f}",
            f"{result.full_dense_s * 1e3:.2f}",
            f"{result.crop_dense_s * 1e3:.2f}",
            f"{result.direct_rle_s * 1e3:.2f}",
            _fmt_ratio(rle_full),
            _fmt_ratio(rle_crop),
            f"{result.rle_runs}/{result.rle_spans}",
            "/".join(
                [
                    _fmt_mb(result.full_dense_peak_bytes),
                    _fmt_mb(result.crop_dense_peak_bytes),
                    _fmt_mb(result.direct_rle_peak_bytes),
                ]
            ),
            f"{_fmt_mb(result.dense_storage_bytes)}/{_fmt_mb(result.compact_storage_bytes)}",
            "[green]✓[/green]" if pixel_perfect else "[red]✗[/red]",
        )
    console.print(table)
    console.print(
        "[dim]"
        + "  ·  ".join(
            [
                f"timings are median of {reps} reps after {warmup} warmups",
                "scratch MB is full/crop/RLE traced allocation while painting "
                "into a preallocated canvas",
                "mask MB is dense/compact persistent mask storage",
                "RLE/full = full dense paint time / direct RLE paint time",
                "RLE/crop = crop-dense paint time / direct RLE paint time",
                "runs/spans exposes Python direct-paint loop count; many spans "
                "can be slower than crop-dense NumPy indexing",
                "OK means crop-dense and direct-RLE outputs exactly match full dense",
            ]
        )
        + "[/dim]"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-segments", type=int, default=MAX_SEGMENTS_PER_IMAGE)
    parser.add_argument("--max-dimension", type=int, default=MAX_IMAGE_DIMENSION)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run synthetic polygon scenarios instead of image-backed scenarios.",
    )
    return parser.parse_args()


def synthetic_inputs() -> list[PaintInput]:
    """Return fallback synthetic benchmark inputs."""
    scenarios = [
        Scenario("720p sparse", (720, 1280), 80, 0.025, 12),
        Scenario("1080p medium", (1080, 1920), 120, 0.050, 24),
        Scenario("1080p complex", (1080, 1920), 120, 0.050, 96),
    ]
    return [make_synthetic_input(scenario) for scenario in scenarios]


def image_inputs(
    source_dir: Path,
    max_images: int | None,
    max_segments: int,
    max_dimension: int,
) -> list[PaintInput]:
    """Load image-backed benchmark inputs with progress reporting."""
    paths = image_paths(source_dir, max_images=max_images)
    inputs: list[PaintInput] = []
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("preparing images...", total=len(paths))
        for path in paths:
            progress.update(task, description=f"[bold]{path.stem}[/bold]")
            try:
                inputs.append(
                    make_image_input(
                        path=path,
                        max_segments=max_segments,
                        max_dimension=max_dimension,
                    )
                )
            except ValueError as exc:
                console.print(f"[yellow]skipped[/yellow] {path.name}: {exc}")
            progress.advance(task)
    return inputs


def main() -> None:
    """Run the benchmark."""
    args = parse_args()
    max_images = None if args.max_images <= 0 else args.max_images
    if args.synthetic:
        inputs = synthetic_inputs()
    else:
        inputs = image_inputs(
            source_dir=args.source_dir,
            max_images=max_images,
            max_segments=args.max_segments,
            max_dimension=args.max_dimension,
        )
        if not inputs:
            console.print(
                f"[yellow]no image inputs found in {args.source_dir}; "
                "falling back to synthetic scenarios[/yellow]"
            )
            inputs = synthetic_inputs()

    results: list[PaintBenchmarkResult] = []
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("benchmarking...", total=len(inputs))
        for paint_input in inputs:
            progress.update(task, description=f"[bold]{paint_input.name}[/bold]")
            results.append(
                run_input(
                    paint_input=paint_input,
                    repetitions=args.repetitions,
                    warmup=args.warmup,
                )
            )
            gc.collect()
            progress.advance(task)

    print_summary(results, reps=args.repetitions, warmup=args.warmup)


if __name__ == "__main__":
    main()
