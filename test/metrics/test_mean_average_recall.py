from contextlib import AbstractContextManager, ExitStack
from typing import Any

import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.metrics import MeanAverageRecall, MetricTarget

# Totals:
# class 0 GT count = 17
# class 1 GT count = 19

TARGETS = [
    # img 0 (2 GT: c0, c1)
    np.array(
        [
            [100, 120, 260, 400, 1.0, 0],
            [500, 200, 760, 640, 1.0, 1],
        ],
        dtype=np.float32,
    ),
    # img 1 (3 GT: c0, c0, c1)
    np.array(
        [
            [50, 60, 180, 300, 1.0, 0],
            [210, 70, 340, 310, 1.0, 0],
            [400, 90, 620, 360, 1.0, 1],
        ],
        dtype=np.float32,
    ),
    # img 2 (1 GT: c1)
    np.array(
        [
            [320, 200, 540, 520, 1.0, 1],
        ],
        dtype=np.float32,
    ),
    # img 3 (4 GT: c0, c1, c0, c1)
    np.array(
        [
            [100, 100, 240, 340, 1.0, 0],
            [260, 110, 410, 350, 1.0, 1],
            [430, 120, 580, 360, 1.0, 0],
            [600, 130, 760, 370, 1.0, 1],
        ],
        dtype=np.float32,
    ),
    # img 4 (2 GT: c0, c0)
    np.array(
        [
            [120, 400, 260, 700, 1.0, 0],
            [300, 420, 480, 720, 1.0, 0],
        ],
        dtype=np.float32,
    ),
    # img 5 (3 GT: c1, c1, c1)
    np.array(
        [
            [50, 50, 200, 260, 1.0, 1],
            [230, 60, 380, 270, 1.0, 1],
            [410, 70, 560, 280, 1.0, 1],
        ],
        dtype=np.float32,
    ),
    # img 6 (1 GT: c0)
    np.array(
        [
            [600, 60, 780, 300, 1.0, 0],
        ],
        dtype=np.float32,
    ),
    # img 7 (5 GT: c0, c1, c1, c0, c1)
    np.array(
        [
            [60, 360, 180, 600, 1.0, 0],
            [200, 350, 340, 590, 1.0, 1],
            [360, 340, 500, 580, 1.0, 1],
            [520, 330, 660, 570, 1.0, 0],
            [680, 320, 820, 560, 1.0, 1],
        ],
        dtype=np.float32,
    ),
    # img 8 (2 GT: c1, c1)
    np.array(
        [
            [100, 100, 220, 300, 1.0, 1],
            [260, 110, 380, 310, 1.0, 1],
        ],
        dtype=np.float32,
    ),
    # img 9 (1 GT: c0)
    np.array(
        [
            [420, 400, 600, 700, 1.0, 0],
        ],
        dtype=np.float32,
    ),
    # img 10 (4 GT: c0, c1, c1, c0)
    np.array(
        [
            [50, 500, 180, 760, 1.0, 0],
            [200, 500, 350, 760, 1.0, 1],
            [370, 500, 520, 760, 1.0, 1],
            [540, 500, 690, 760, 1.0, 0],
        ],
        dtype=np.float32,
    ),
    # img 11 (2 GT: c1, c0)
    np.array(
        [
            [150, 150, 300, 420, 1.0, 1],
            [330, 160, 480, 430, 1.0, 0],
        ],
        dtype=np.float32,
    ),
    # img 12 (3 GT: c0, c1, c1)
    np.array(
        [
            [600, 200, 760, 460, 1.0, 0],
            [100, 220, 240, 480, 1.0, 1],
            [260, 230, 400, 490, 1.0, 1],
        ],
        dtype=np.float32,
    ),
    # img 13 (1 GT: c0)
    np.array(
        [
            [50, 50, 190, 250, 1.0, 0],
        ],
        dtype=np.float32,
    ),
    # img 14 (2 GT: c1, c0)
    np.array(
        [
            [420, 80, 560, 300, 1.0, 1],
            [580, 90, 730, 310, 1.0, 0],
        ],
        dtype=np.float32,
    ),
]

PREDICTIONS = [
    # img 0: 2 TP + 1 class mismatch FP
    np.array(
        [
            [102, 118, 258, 398, 0.94, 0],  # TP (c0)
            [500, 200, 760, 640, 0.90, 1],  # TP (c1)
            [100, 120, 260, 400, 0.55, 1],  # FP (class mismatch)
        ],
        dtype=np.float32,
    ),
    # img 1: TPs for two c0, miss c1 (FN) + background FP
    np.array(
        [
            [50, 60, 180, 300, 0.91, 0],  # TP (c0)
            [210, 70, 340, 310, 0.88, 0],  # TP (c0)
            [600, 400, 720, 560, 0.42, 1],  # FP (no GT nearby)
        ],
        dtype=np.float32,
    ),
    # img 2: Low-IoU (miss) + random FP
    np.array(
        [
            [300, 180, 500, 430, 0.83, 1],  # Low IoU (shifted, suppose < threshold)
            [50, 50, 140, 140, 0.30, 0],  # FP
        ],
        dtype=np.float32,
    ),
    # img 3: Only match two (others FN) + one mismatch
    np.array(
        [
            [100, 100, 240, 340, 0.90, 0],  # TP (c0)
            [260, 110, 410, 350, 0.87, 1],  # TP (c1)
            [430, 120, 580, 360, 0.70, 1],  # FP (class mismatch; GT is c0)
        ],
        dtype=np.float32,
    ),
    # img 4: No predictions (2 FN)
    np.array([], dtype=np.float32).reshape(0, 6),
    # img 5: All three matched + class mismatch
    np.array(
        [
            [50, 50, 200, 260, 0.95, 1],  # TP (c1)
            [230, 60, 380, 270, 0.92, 1],  # TP (c1)
            [410, 70, 560, 280, 0.90, 1],  # TP (c1)
            [50, 50, 200, 260, 0.40, 0],  # FP (class mismatch)
        ],
        dtype=np.float32,
    ),
    # img 6: Wrong class over GT (0 recall)
    np.array(
        [
            [600, 60, 780, 300, 0.89, 1],  # FP (class mismatch)
        ],
        dtype=np.float32,
    ),
    # img 7: 3 TP, 1 miss (only 3/5 recalled)
    np.array(
        [
            [60, 360, 180, 600, 0.93, 0],  # TP (c0)
            [200, 350, 340, 590, 0.90, 1],  # TP (c1)
            [360, 340, 500, 580, 0.88, 1],  # TP (c1)
            [520, 330, 660, 570, 0.50, 1],  # FP (class mismatch; GT is c0)
        ],
        dtype=np.float32,
    ),
    # img 8: 2 TP
    np.array(
        [
            [100, 100, 220, 300, 0.96, 1],  # TP
            [262, 112, 378, 308, 0.89, 1],  # TP
        ],
        dtype=np.float32,
    ),
    # img 9: 1 TP + 1 FP
    np.array(
        [
            [418, 398, 602, 702, 0.86, 0],  # TP
            [100, 100, 140, 160, 0.33, 1],  # FP
        ],
        dtype=np.float32,
    ),
    # img 10: Perfect (all 4 TP)
    np.array(
        [
            [50, 500, 180, 760, 0.94, 0],  # TP
            [200, 500, 350, 760, 0.93, 1],  # TP
            [370, 500, 520, 760, 0.92, 1],  # TP
            [540, 500, 690, 760, 0.91, 0],  # TP
        ],
        dtype=np.float32,
    ),
    # img 11: 1 TP, 1 low IoU (FN remains) + FP
    np.array(
        [
            [150, 150, 300, 420, 0.90, 1],  # TP (c1)
            [
                332,
                162,
                478,
                428,
                0.58,
                0,
            ],  # TP? (slight shift) treat as TP if IoU high enough; assume OK
            [148, 148, 298, 415, 0.52, 0],  # FP (class mismatch over c1)
        ],
        dtype=np.float32,
    ),
    # img 12: 2 TP + 1 miss (one c1 missed)
    np.array(
        [
            [600, 200, 760, 460, 0.92, 0],  # TP
            [100, 220, 240, 480, 0.90, 1],  # TP
            [260, 230, 400, 490, 0.40, 0],  # FP (class mismatch; GT is c1)
        ],
        dtype=np.float32,
    ),
    # img 13: No predictions (1 FN)
    np.array([], dtype=np.float32).reshape(0, 6),
    # img 14: Class swapped (0 recall) + one correct + one FP
    np.array(
        [
            [420, 80, 560, 300, 0.88, 0],  # FP (class mismatch; GT is c1)
            [580, 90, 730, 310, 0.86, 1],  # FP (class mismatch; GT is c0)
        ],
        dtype=np.float32,
    ),
]


# Expected mAR at K = 1, 10, 100
EXPECTED_RESULT = np.array([0.2874613, 0.63622291, 0.63622291])


def mock_detections_list(boxes_list):
    return [
        Detections(
            xyxy=boxes[:, :4], confidence=boxes[:, 4], class_id=boxes[:, 5].astype(int)
        )
        for boxes in boxes_list
    ]


@pytest.mark.parametrize(
    "predictions_list, targets_list, expected_result, exception",
    [
        (
            mock_detections_list(PREDICTIONS),
            mock_detections_list(TARGETS),
            EXPECTED_RESULT,
            ExitStack(),
        ),
    ],
)
def test_recall(
    predictions_list: list[Detections],
    targets_list: list[Detections],
    expected_result: np.ndarray,
    exception: AbstractContextManager[Any],
):
    mar_metrics = MeanAverageRecall(metric_target=MetricTarget.BOXES)
    mar_result = mar_metrics._compute(predictions_list, targets_list)

    with exception:
        np.testing.assert_almost_equal(
            mar_result.recall_scores, expected_result, decimal=5
        )
