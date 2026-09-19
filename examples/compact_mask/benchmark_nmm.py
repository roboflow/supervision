"""Measure compact NMM with fixed mask crops and varying canvas/fragmentation.

Run from the checkout being measured (NumPy and supervision are sufficient)::

    python examples/compact_mask/benchmark_nmm.py --canvas 512 2048 4096

Compare the same command on the base and proposed revisions. Input construction is
excluded from measurements. Python and NumPy allocations are measured by tracemalloc;
this is peak traced allocation during NMM, not process RSS or GPU memory. Each JSON line
reports median uninstrumented wall time and maximum traced peak from separate calls over
the requested repeats, so tracing overhead is excluded from the timing.
"""

import argparse
import gc
import json
import statistics
import time
import tracemalloc

import numpy as np

import supervision as sv
from supervision.detection.compact_mask import CompactMask


def make_detections(canvas: int, crop_size: int, pattern: str) -> sv.Detections:
    """Place four groups of three duplicate crops onto a logical square canvas."""
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
    for group in range(4):
        moved = local.with_offset(group * (crop_size + 8), 10, (canvas, canvas))
        masks.extend([moved] * 3)
    compact = CompactMask.merge(masks)
    return sv.Detections(
        xyxy=compact.bbox_xyxy.astype(float),
        confidence=np.linspace(0.99, 0.5, len(compact)),
        class_id=np.zeros(len(compact), dtype=int),
        mask=compact,
    )


def measure(detections: sv.Detections, repeats: int) -> dict[str, int | float]:
    """Time public NMM and track temporary allocation without measuring ingestion."""
    durations = []
    peaks = []
    result = detections
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
    if not isinstance(result.mask, CompactMask) or len(result) != 4:
        raise RuntimeError("NMM must return four compact mask groups")
    return {
        "seconds_median": statistics.median(durations),
        "peak_traced_mib": max(peaks) / 1024**2,
        "output_count": len(result),
        "output_area": int(result.mask.area.sum()),
    }


def main() -> None:
    """Print reproducible JSON rows for each canvas and mask complexity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canvas", type=int, nargs="+", default=[512, 2048, 4096])
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--pattern",
        nargs="+",
        choices=["solid", "checkerboard", "random"],
        default=["solid", "checkerboard", "random"],
    )
    args = parser.parse_args()
    if args.crop_size < 1 or args.repeats < 1:
        parser.error("crop-size and repeats must be positive")
    if min(args.canvas) < 4 * (args.crop_size + 8):
        parser.error("each canvas must fit four disjoint crops")
    for canvas in args.canvas:
        for pattern in args.pattern:
            detections = make_detections(canvas, args.crop_size, pattern)
            if not isinstance(detections.mask, CompactMask):
                raise RuntimeError("Benchmark inputs must have compact masks")
            row = {
                "canvas": canvas,
                "crop_size": args.crop_size,
                "pattern": pattern,
                "instances": len(detections),
                "input_rle_counts": sum(len(rle) for rle in detections.mask._rles),
                **measure(detections, args.repeats),
            }
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
