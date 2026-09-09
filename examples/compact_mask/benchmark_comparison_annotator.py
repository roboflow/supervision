"""Benchmark dense-mask union in ``ComparisonAnnotator``.

Purpose:
    Measure the dense-mask reduction changed by PR #2495 against the exact former
    per-mask OR loop. The benchmark isolates the reduction so drawing, color blending,
    and CompactMask crop decoding do not obscure its result.
Scope:
    Dense ``(N, H, W)`` boolean masks only. It does not claim performance for
    CompactMask, GPU execution, or a full annotation pipeline.
Usage:
    Run ``uv run python examples/compact_mask/benchmark_comparison_annotator.py``.
    Use ``--output result.json`` to retain machine-readable environment, workload,
    parity, and timing evidence. Defaults are deterministic and representative of a
    1280x720 frame with 32 masks.
Outputs:
    Prints JSON containing median per-call times, speedup, and both union and public
    annotation parity. When ``--output`` is supplied, writes the same JSON file.
Failure:
    Exits with an assertion error if the old and current reductions or their public
    annotation outputs differ. Invalid CLI values are rejected by ``argparse``.
Used by:
    PR #2495 review remediation and maintainers assessing the vectorized dense-mask
    union. The script has no network, model, image-file, or optional dependency input.
"""

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path
from statistics import median

import numpy as np

import supervision as sv


def parse_args() -> argparse.Namespace:
    """Parse the deterministic dense-mask benchmark configuration."""
    parser = argparse.ArgumentParser(
        description="Benchmark ComparisonAnnotator dense-mask union."
    )
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--detections", type=int, default=32)
    parser.add_argument("--mask-side", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def build_dense_detections(
    height: int, width: int, detection_count: int, mask_side: int, seed: int
) -> sv.Detections:
    """Build deterministic, separated dense masks for the measured reduction."""
    if min(height, width, detection_count, mask_side) < 1:
        raise ValueError(
            "Frame dimensions, detections, and mask side must be positive."
        )
    if mask_side > min(height, width):
        raise ValueError("Mask side must fit within both frame dimensions.")

    rng = np.random.default_rng(seed)
    masks = np.zeros((detection_count, height, width), dtype=bool)
    xyxy = np.empty((detection_count, 4), dtype=np.float32)
    max_x = width - mask_side + 1
    max_y = height - mask_side + 1
    for index in range(detection_count):
        x1 = int(rng.integers(max_x))
        y1 = int(rng.integers(max_y))
        x2 = x1 + mask_side
        y2 = y1 + mask_side
        masks[index, y1:y2, x1:x2] = True
        xyxy[index] = (x1, y1, x2, y2)
    return sv.Detections(xyxy=xyxy, mask=masks)


def legacy_mask_union(scene: np.ndarray, detections: sv.Detections) -> np.ndarray:
    """Reproduce the pre-PR dense-mask OR loop for a fair baseline."""
    mask = np.zeros(scene.shape[:2], dtype=np.bool_)
    if detections.is_empty():
        return mask

    if not isinstance(detections.mask, np.ndarray):
        raise TypeError("The legacy benchmark baseline requires dense NumPy masks.")
    for detection_mask in detections.mask:
        mask |= detection_mask.astype(np.bool_)
    return mask


def current_mask_union(scene: np.ndarray, detections: sv.Detections) -> np.ndarray:
    """Run the current implementation under test on a dense mask stack."""
    return sv.ComparisonAnnotator._mask_from_mask(scene, detections)


def time_mask_union(
    function: Callable[[np.ndarray, sv.Detections], np.ndarray],
    scene: np.ndarray,
    detections: sv.Detections,
    iterations: int,
    repeats: int,
) -> list[float]:
    """Return median-ready per-call nanosecond samples after warmup."""
    if min(iterations, repeats) < 1:
        raise ValueError("Iterations and repeats must be positive.")

    function(scene, detections)
    samples = []
    for _ in range(repeats):
        started_at = time.perf_counter_ns()
        for _ in range(iterations):
            function(scene, detections)
        samples.append((time.perf_counter_ns() - started_at) / iterations)
    return samples


def current_annotation_matches_legacy_union(
    scene: np.ndarray, detections: sv.Detections
) -> bool:
    """Check public annotation output against the exact legacy union result."""
    annotator = sv.ComparisonAnnotator(opacity=1.0)
    result = annotator.annotate(
        scene=scene.copy(),
        detections_1=detections,
        detections_2=sv.Detections.empty(),
    )
    expected = scene.copy()
    expected[legacy_mask_union(scene, detections)] = annotator.color_1.as_bgr()
    return bool(np.array_equal(result, expected))


def main() -> None:
    """Run the parity checks and emit reproducible legacy/current timing evidence."""
    args = parse_args()
    scene = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    detections = build_dense_detections(
        height=args.height,
        width=args.width,
        detection_count=args.detections,
        mask_side=args.mask_side,
        seed=args.seed,
    )
    legacy_union = legacy_mask_union(scene, detections)
    current_union = current_mask_union(scene, detections)
    union_parity = bool(np.array_equal(legacy_union, current_union))
    annotation_parity = current_annotation_matches_legacy_union(scene, detections)
    if not union_parity or not annotation_parity:
        raise AssertionError(
            "Current dense-mask union does not match the legacy output."
        )

    legacy_samples_ns = time_mask_union(
        legacy_mask_union, scene, detections, args.iterations, args.repeats
    )
    current_samples_ns = time_mask_union(
        current_mask_union, scene, detections, args.iterations, args.repeats
    )
    legacy_median_ns = median(legacy_samples_ns)
    current_median_ns = median(current_samples_ns)
    result = {
        "workload": {
            "height": args.height,
            "width": args.width,
            "dense_masks": args.detections,
            "mask_side": args.mask_side,
            "iterations_per_repeat": args.iterations,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "timing_method": "perf_counter_ns; warmup once; median per-call nanoseconds",
        "parity": {
            "dense_union": union_parity,
            "public_annotation": annotation_parity,
        },
        "legacy_median_ms": legacy_median_ns / 1_000_000,
        "current_median_ms": current_median_ns / 1_000_000,
        "speedup": legacy_median_ns / current_median_ns,
    }
    rendered_result = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered_result, end="")
    if args.output is not None:
        args.output.write_text(rendered_result)


if __name__ == "__main__":
    main()
