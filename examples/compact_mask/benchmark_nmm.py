"""Measure compact NMM with fixed mask crops and varying canvas/fragmentation.

Run from the checkout being measured (NumPy and supervision are sufficient)::

    python examples/compact_mask/benchmark_nmm.py --canvas 512 2048 4096
    python examples/compact_mask/benchmark_nmm.py --canvas 512 --crop-size 200 \\
        --pattern checkerboard
    python examples/compact_mask/benchmark_nmm.py --canvas 512 --crop-size 200 \\
        --pattern solid

Compare the same command on the base and proposed revisions. Input construction is
excluded from measurements. Python and NumPy allocations are measured by tracemalloc;
this is peak traced allocation during NMM, not process RSS or GPU memory. Each table row
reports median uninstrumented wall time and maximum traced peak from separate calls over
the requested repeats, so tracing overhead is excluded from the timing.

The 200-pixel checkerboard case is intentionally fragmented and exposes RLE-union
regressions that the default 64-pixel canvas sweep can hide. The matching solid case
measures a large, clean mask with the same crop and canvas dimensions.
"""

import argparse
import gc
import statistics
import time
import tracemalloc

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table

import supervision as sv
from supervision.detection.compact_mask import CompactMask

console = Console(width=140, force_terminal=True)

GROUP_COUNT = 4


def make_detections(
    canvas: int, crop_size: int, pattern: str, duplicates: int
) -> sv.Detections:
    """Place four disjoint groups of duplicate crops onto a logical square canvas."""
    rng = np.random.default_rng(20260919)
    crop = np.ones((crop_size, crop_size), dtype=bool)
    if pattern == "checkerboard":
        crop = np.indices(crop.shape).sum(axis=0) % 2 == 0
    elif pattern == "random":
        crop = rng.random(crop.shape) > 0.5
    local = CompactMask.from_dense(
        crop[None],
        np.array([[0, 0, crop_size - 1, crop_size - 1]]),
        (crop_size, crop_size),
    )
    masks = []
    crop_stride = crop_size + 8
    for group in range(GROUP_COUNT):
        row, column = divmod(group, 2)
        moved = local.with_offset(
            column * crop_stride, row * crop_stride, (canvas, canvas)
        )
        masks.extend([moved] * duplicates)
    compact = CompactMask.merge(masks)
    return sv.Detections(
        xyxy=compact.bbox_xyxy.astype(float),
        confidence=np.linspace(0.99, 0.5, len(compact)),
        class_id=np.zeros(len(compact), dtype=int),
        mask=compact,
    )


def measure(
    detections: sv.Detections, repeats: int, expected_output_area: int
) -> dict[str, int | float]:
    """Time public NMM and track temporary allocation without measuring ingestion."""
    durations = []
    peaks = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        result = detections.with_nmm(threshold=0.5)
        durations.append(time.perf_counter() - start)
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        result = detections.with_nmm(threshold=0.5)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)
    if not isinstance(result.mask, CompactMask) or len(result) != GROUP_COUNT:
        raise RuntimeError(f"NMM must return {GROUP_COUNT} compact mask groups")
    output_area = int(result.mask.area.sum())
    if output_area != expected_output_area:
        raise RuntimeError(
            "NMM output area does not equal the area of one crop per duplicate group"
        )
    return {
        "seconds_median": statistics.median(durations),
        "peak_traced_mib": max(peaks) / 1024**2,
        "output_count": len(result),
        "output_area": output_area,
    }


def build_table(rows: list[dict[str, int | float | str]]) -> Table:
    """Format benchmark rows as a rich table matching the sibling scripts' style."""
    table = Table(
        title="Compact NMM union benchmark",
        box=box.ROUNDED,
        header_style="bold",
    )
    table.add_column("Canvas", justify="right")
    table.add_column("Crop", justify="right")
    table.add_column("Pattern")
    table.add_column("Dup.", justify="right")
    table.add_column("Instances", justify="right")
    table.add_column("Input\nRLE runs", justify="right")
    table.add_column("Output\ncount", justify="right", style="green")
    table.add_column("Output\narea", justify="right", style="green")
    table.add_column("Time\n(median, ms)", justify="right", style="yellow")
    table.add_column("Peak traced\n(MiB)", justify="right", style="cyan")
    for row in rows:
        table.add_row(
            str(row["canvas"]),
            str(row["crop_size"]),
            str(row["pattern"]),
            str(row["duplicates"]),
            str(row["instances"]),
            str(row["input_rle_counts"]),
            str(row["output_count"]),
            str(row["output_area"]),
            f"{row['seconds_median'] * 1e3:.3f}",
            f"{row['peak_traced_mib']:.3f}",
        )
    return table


def main() -> None:
    """Print a reproducible rich table row for each canvas and mask complexity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canvas", type=int, nargs="+", default=[512, 2048, 4096])
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--duplicates",
        type=int,
        default=3,
        help="identical detections in each of the four NMM groups (default: 3)",
    )
    parser.add_argument(
        "--pattern",
        nargs="+",
        choices=["solid", "checkerboard", "random"],
        default=["solid", "checkerboard", "random"],
    )
    args = parser.parse_args()
    if args.crop_size < 1 or args.repeats < 1 or args.duplicates < 1:
        parser.error("crop-size, repeats, and duplicates must be positive")
    if min(args.canvas) < 2 * args.crop_size + 8:
        parser.error("each canvas must fit four disjoint crops in a 2x2 grid")
    rows = []
    for canvas in args.canvas:
        for pattern in args.pattern:
            detections = make_detections(
                canvas, args.crop_size, pattern, args.duplicates
            )
            if not isinstance(detections.mask, CompactMask):
                raise RuntimeError("Benchmark inputs must have compact masks")
            expected_output_area = int(detections.mask.area.sum() // args.duplicates)
            rows.append(
                {
                    "canvas": canvas,
                    "crop_size": args.crop_size,
                    "pattern": pattern,
                    "duplicates": args.duplicates,
                    "instances": len(detections),
                    "input_rle_counts": sum(len(rle) for rle in detections.mask._rles),
                    "expected_output_area": expected_output_area,
                    **measure(detections, args.repeats, expected_output_area),
                }
            )
    console.print(build_table(rows))


if __name__ == "__main__":
    main()
