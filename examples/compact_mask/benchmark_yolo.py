"""Measure YOLO loading time, mask buffers and peak RSS on Linux.

Run ``uv run python examples/compact_mask/benchmark_yolo.py`` from the repository
root. Each mode runs three times in fresh processes; input generation and imports
are outside the loader timing. Peak RSS includes process/import overhead and
input generation. Mask buffer bytes exclude Python objects and allocator overhead.
Pixel equality is checked separately so decoding does not inflate measured RSS.

All inputs are synthetic: 16 images with 8 small polygons each. Compression and
timing will differ for real datasets, especially fragmented masks. No model,
dataset download, GPU, or additional dependencies are needed.
"""

import argparse
import json
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

import supervision as sv


def worker(mode: str, scenario: str) -> None:
    """Build identical datasets outside the loader timing and report mask buffers."""
    width, height = (1024, 1024) if scenario != "wide" else (1920, 1080)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        images, labels = root / "images", root / "labels"
        images.mkdir()
        labels.mkdir()
        rows = []
        for j in range(8):
            x = 0.05 + j * 0.1
            if scenario == "rectangles":
                points = [(x, 0.1), (x + 0.04, 0.1), (x + 0.04, 0.15), (x, 0.15)]
            else:
                points = [(x, 0.1), (x + 0.04, 0.1), (x + 0.025, 0.15)]
            rows.append("0 " + " ".join(str(c) for point in points for c in point))
        for i in range(16):
            Image.new("RGB", (width, height)).save(images / f"{i:03}.png")
            (labels / f"{i:03}.txt").write_text("\n".join(rows) + "\n")
        (root / "data.yaml").write_text("names: [object]\n")
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        start = perf_counter()
        dataset = sv.DetectionDataset.from_yolo(
            str(images),
            str(labels),
            str(root / "data.yaml"),
            compact_masks=mode == "compact",
        )
        elapsed = perf_counter() - start
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if mode == "verify":
            compact = sv.DetectionDataset.from_yolo(
                str(images), str(labels), str(root / "data.yaml"), compact_masks=True
            )
            for path, expected in dataset.annotations.items():
                actual = compact.annotations[path]
                if not isinstance(actual.mask, sv.CompactMask):
                    raise TypeError("Compact loading did not return CompactMask")
                np.testing.assert_array_equal(actual.mask.to_dense(), expected.mask)
                np.testing.assert_array_equal(actual.xyxy, expected.xyxy)
                np.testing.assert_array_equal(actual.class_id, expected.class_id)
            print(json.dumps({"scenario": scenario, "pixel_equality": True}))
            return
        buffers = 0
        for detections in dataset.annotations.values():
            mask = detections.mask
            if isinstance(mask, sv.CompactMask):
                buffers += mask._offsets.nbytes + mask._crop_shapes.nbytes
                buffers += sum(rle.nbytes for rle in mask._rles)
            else:
                if not isinstance(mask, np.ndarray):
                    raise TypeError("Dense loading did not return a NumPy mask")
                buffers += mask.nbytes
        print(
            json.dumps(
                {
                    "mode": mode,
                    "scenario": scenario,
                    "resolution": [width, height],
                    "mask_buffer_bytes": buffers,
                    "load_seconds": elapsed,
                    "peak_rss_mib": rss_after / 1024,
                    "rss_before_mib": rss_before / 1024,
                }
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dense", "compact", "verify"])
    parser.add_argument(
        "--scenario", choices=["rectangles", "triangles", "wide"], default="triangles"
    )
    args = parser.parse_args()
    if args.mode:
        worker(args.mode, args.scenario)
    else:
        for scenario in ("rectangles", "triangles", "wide"):
            for mode in ("dense", "compact"):
                samples = []
                for _ in range(3):
                    # Only this interpreter/file and fixed mode/scenario choices run.
                    result = subprocess.run(  # noqa: S603
                        [
                            sys.executable,
                            __file__,
                            "--mode",
                            mode,
                            "--scenario",
                            scenario,
                        ],
                        text=True,
                        capture_output=True,
                        check=True,
                        timeout=120,
                    )
                    samples.append(json.loads(result.stdout))
                print(
                    json.dumps(
                        {"scenario": scenario, "mode": mode, "samples": samples}
                    ),
                    flush=True,
                )
            # Use the same trusted command for an untimed pixel comparison.
            verification = subprocess.run(  # noqa: S603
                [sys.executable, __file__, "--mode", "verify", "--scenario", scenario],
                text=True,
                capture_output=True,
                check=True,
                timeout=120,
            )
            print(verification.stdout, end="", flush=True)
