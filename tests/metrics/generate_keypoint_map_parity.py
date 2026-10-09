"""Regenerate the `pycocotools` parity constants of the keypoint mAP tests.

`EXPECTED_STATS` and `EXPECTED_AP_PER_CLASS` in the keypoint mAP test module are the
results of `COCOeval(..., iouType="keypoints")` on `_make_synthetic_pose_images()`.
`pycocotools` is not a dependency of the repository, so this script, which pytest does
not collect as a test, rebuilds them on demand. Run it from the repository root whenever
the synthetic data changes, for example when the RNG call order in
`_make_synthetic_pose_images` is edited, and paste the printed constants into the test
module:

    uv run --with pycocotools python -m tests.metrics.generate_keypoint_map_parity

The script also prints how far the stored constants are from the fresh results.
"""

from __future__ import annotations

from importlib.metadata import version
from typing import Any

import numpy as np

from tests.metrics.test_keypoint_mean_average_precision import (
    EXPECTED_AP_PER_CLASS,
    EXPECTED_STATS,
    SyntheticPoseImage,
    _make_synthetic_pose_images,
)


def _flat_keypoints(xy: np.ndarray, visibility: np.ndarray) -> list[float]:
    """Flatten `(K, 2)` coordinates and `(K,)` flags into COCO `[x, y, v, ...]`."""
    return [float(value) for (x, y), v in zip(xy, visibility) for value in (x, y, v)]


def _keypoints_span(xy: np.ndarray) -> list[float]:
    """Return the `[x, y, width, height]` box spanning `(K, 2)` coordinates."""
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def _to_coco(
    images: list[SyntheticPoseImage],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Convert synthetic images to a COCO keypoint dataset and a result list.

    Target points carry `v=2` when visible and `v=0` otherwise, so a target without
    visible points has `num_keypoints=0` and is an ignore region. Predicted points
    carry `v=1`, which `computeOks` does not read. Target areas come from
    `targets.data["area"]`, as in the test.
    """
    annotations: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for index, image in enumerate(images):
        image_id = index + 1  # ids start at 1, since 0 means "no match"
        targets, predictions = image.targets, image.predictions
        for target_index in range(len(targets)):
            xy = targets.xy[target_index].astype(np.float64)
            visible = np.asarray(targets.visible)[target_index]
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": int(targets.class_id[target_index]),
                    "iscrowd": 0,
                    "area": float(targets.data["area"][target_index]),
                    "bbox": _keypoints_span(xy),
                    "num_keypoints": int(visible.sum()),
                    "keypoints": _flat_keypoints(xy, 2 * visible.astype(int)),
                }
            )
        for pred_index in range(len(predictions)):
            xy = predictions.xy[pred_index].astype(np.float64)
            results.append(
                {
                    "image_id": image_id,
                    "category_id": int(predictions.class_id[pred_index]),
                    "score": float(predictions.detection_confidence[pred_index]),
                    "keypoints": _flat_keypoints(xy, np.ones(len(xy), dtype=int)),
                }
            )
    categories = [{"id": class_id, "name": str(class_id)} for class_id in (0, 1)]
    dataset = {
        "images": [{"id": index + 1} for index in range(len(images))],
        "annotations": annotations,
        "categories": categories,
    }
    return dataset, results


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
