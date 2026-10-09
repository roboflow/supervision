"""Regenerate the `pycocotools` parity constants of the keypoint mAP tests.

`EXPECTED_STATS` and `EXPECTED_AP_PER_CLASS` in the keypoint mAP test module are the
results of `COCOeval(..., iouType="keypoints")` on `_make_synthetic_pose_images()`.
`pycocotools` is not a dependency of the repository, so this script, which pytest does
not collect as a test, rebuilds them on demand. Run it from the repository root whenever
the synthetic data changes, for example when the RNG call order in
`_make_synthetic_pose_images` is edited, and paste the printed constants into the test
module:

    uv run --with pycocotools python -m tests.metrics._generate_keypoint_map_parity

The script also prints how far the stored constants are from the fresh results.
"""

from __future__ import annotations

from importlib.metadata import version

import numpy as np

from tests.metrics.test_keypoint_mean_average_precision import (
    EXPECTED_AP_PER_CLASS,
    EXPECTED_STATS,
    _make_synthetic_pose_images,
    _to_coco,
)


def main() -> None:
    """Run `COCOeval` on the synthetic data and print the parity constants."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    dataset, results = _to_coco(_make_synthetic_pose_images())
    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="keypoints")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = [float(value) for value in evaluator.stats[:5]]
    # `eval["precision"]` is (OKS thresholds, recall, category, area, maxDets); take
    # area "all" and the only maxDets, average over recall, order as (category, OKS).
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    ap_per_class = precision.mean(axis=1).T

    print(f"pycocotools {version('pycocotools')}")
    print(f"EXPECTED_STATS = {stats!r}")
    print(f"EXPECTED_AP_PER_CLASS = {ap_per_class.tolist()!r}")
    print(f"max |stats - stored|: {np.abs(stats - np.array(EXPECTED_STATS)).max():.3g}")
    print(
        "max |ap_per_class - stored|: "
        f"{np.abs(ap_per_class - EXPECTED_AP_PER_CLASS).max():.3g}"
    )


if __name__ == "__main__":
    main()
