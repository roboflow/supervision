from contextlib import ExitStack as DoesNotRaise
from test.test_utils import assert_almost_equal, mock_detections
from typing import List

import numpy as np
import pytest

from supervision import ByteTrack
from supervision.detection.core import Detections
from supervision.tracker.byte_tracker.core import STrack

BYTE_TRACKER = ByteTrack(lost_track_buffer=30)

IMG_HEIGHT = 1280
IMG_WIDTH = 2560

# 0:4 bbox, 4 confidence, 5 class_id, 6 tracker_id
PREDICTIONS = np.array(
    [
        [2254, 906, 2447, 1353, 0.90538, 30, 39],
        [2049, 1133, 2226, 1371, 0.96, 56, 40],
        [727, 1224, 838, 1601, 0.72, 39, 41],
        [808, 1214, 910, 1564, 0.89, 39, 42],
        [6, 52, 1131, 2133, 0.94, 72, 43],
        [299, 1225, 512, 1663, 0.73, 4, 44],
        [529, 874, 645, 945, 0.76, 3, 45],
        [8, 47, 1935, 2135, 0.95, 72, 46],
        [2265, 813, 2328, 901, 0.84, 62, 47],
    ],
    dtype=np.float32,
)

TARGET_TRACKER_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)

LOW_CONFIDENCE_PREDICTIONS = PREDICTIONS.copy()
# all low confidence predictions (<.1)
LOW_CONFIDENCE_PREDICTIONS[:, 4] = 0.1

# Test data value
DATA = {
    "number": [5 * i for i in range(PREDICTIONS.shape[0])],
    "letter": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
}

TARGET_PREDICTIONS = PREDICTIONS.copy()
# Add 1 pixel to every dimension every frame.
TARGET_PREDICTIONS[:, :4] += 3

LOW_CONFIDENCE_TARGET_PREDICTIONS = TARGET_PREDICTIONS.copy()
# all low confidence predictions (<.1)
LOW_CONFIDENCE_TARGET_PREDICTIONS[:, 4] = 0.1

# set mask to only be inside bboxes
TARGET_MASK = np.zeros((TARGET_PREDICTIONS.shape[0], IMG_HEIGHT, IMG_WIDTH))
for i in range(TARGET_PREDICTIONS.shape[0]):
    #   box            y                        x
    TARGET_MASK[
        i,  # Box index
        int(TARGET_PREDICTIONS[i, 1]) : int(TARGET_PREDICTIONS[i, 3]),  # Y range
        int(TARGET_PREDICTIONS[i, 0]) : int(TARGET_PREDICTIONS[i, 2]),  # X range
    ] = 1

TARGET_EMPTY_DETECTIONS = Detections.empty()
TARGET_EMPTY_DETECTIONS.tracker_id = np.ndarray(shape=(0), dtype=np.int64)


def assert_strack_lists_equal(stracks, target_stracks):
    assert len(stracks) == len(
        target_stracks
    ), f"Expected {len(target_stracks)} stracks, but got {len(stracks)}."
    for i in range(len(stracks)):
        strack = stracks[i]
        target_strack = target_stracks[i]
        for j in range(strack.tlwh.shape[0]):
            # check if the boxes are within 3 pixels of each other
            assert_almost_equal(strack.tlwh[j], target_strack.tlwh[j], tolerance=3)
        assert (
            strack.start_frame == target_strack.start_frame
        ), f"Expected start frame {strack.start_frame},\
             but got {target_strack.start_frame}."
        assert (
            strack.frame_id == target_strack.frame_id
        ), f"Expected current frame {strack.frame_id},\
             but got {target_strack.frame_id}."


@pytest.mark.parametrize(
    "incoming_detections, target_detections, with_mask," " exception",
    [
        (  # Test empty detections
            Detections.empty(),
            TARGET_EMPTY_DETECTIONS,  # empty detections with tracker_id array
            False,
            DoesNotRaise(),
        ),
        (  # Test base detections required for tracker
            mock_detections(
                xyxy=PREDICTIONS[:, :4],
                confidence=PREDICTIONS[:, 4],
                class_id=PREDICTIONS[:, 5].astype(int),
            ),
            mock_detections(
                xyxy=TARGET_PREDICTIONS[:, :4],
                confidence=TARGET_PREDICTIONS[:, 4],
                class_id=TARGET_PREDICTIONS[:, 5].astype(int),
                tracker_id=TARGET_TRACKER_IDS,
            ),
            False,
            DoesNotRaise(),
        ),
        (  # Test base detections with low confidence
            mock_detections(
                xyxy=LOW_CONFIDENCE_PREDICTIONS[:, :4],
                confidence=LOW_CONFIDENCE_PREDICTIONS[:, 4],  # confidence < .1
                class_id=LOW_CONFIDENCE_PREDICTIONS[:, 5].astype(int),
            ),
            TARGET_EMPTY_DETECTIONS,
            False,
            DoesNotRaise(),
        ),
        (  # Test segmentation detections
            mock_detections(
                xyxy=PREDICTIONS[:, :4],
                confidence=PREDICTIONS[:, 4],
                class_id=PREDICTIONS[:, 5].astype(int),
            ),
            # Expect detections with mask
            mock_detections(
                xyxy=TARGET_PREDICTIONS[:, :4],
                confidence=TARGET_PREDICTIONS[:, 4],
                class_id=TARGET_PREDICTIONS[:, 5].astype(int),
                mask=TARGET_MASK,
                tracker_id=TARGET_TRACKER_IDS,
            ),
            True,
            DoesNotRaise(),
        ),
        (  # Test segmentation detections with existing tracker_id's
            mock_detections(
                xyxy=PREDICTIONS[:, :4],
                confidence=PREDICTIONS[:, 4],
                class_id=PREDICTIONS[:, 5].astype(int),
                tracker_id=PREDICTIONS[:, 6].astype(int),
            ),
            mock_detections(
                xyxy=TARGET_PREDICTIONS[:, :4],
                confidence=TARGET_PREDICTIONS[:, 4],
                class_id=TARGET_PREDICTIONS[:, 5].astype(int),
                tracker_id=TARGET_TRACKER_IDS,
                mask=TARGET_MASK,
            ),
            True,
            DoesNotRaise(),
        ),
        (  # Test segmentation detections with data argument added
            mock_detections(
                xyxy=PREDICTIONS[:, :4],
                confidence=PREDICTIONS[:, 4],
                class_id=PREDICTIONS[:, 5].astype(int),
                tracker_id=PREDICTIONS[:, 6].astype(int),
                data=DATA,
            ),
            mock_detections(
                xyxy=TARGET_PREDICTIONS[:, :4],
                confidence=TARGET_PREDICTIONS[:, 4],
                class_id=TARGET_PREDICTIONS[:, 5].astype(int),
                tracker_id=TARGET_TRACKER_IDS,
                mask=TARGET_MASK,
                data=DATA,
            ),
            True,
            DoesNotRaise(),
        ),
    ],
)
def test_update_with_detections(
    incoming_detections: Detections,
    target_detections: Detections,
    with_mask: bool,
    exception: Exception,
):
    incoming_detections.xyxy = (
        incoming_detections.xyxy.copy()
        if incoming_detections.xyxy is not None
        else None
    )
    incoming_detections.confidence = (
        incoming_detections.confidence.copy()
        if incoming_detections.confidence is not None
        else None
    )
    incoming_detections.class_id = (
        incoming_detections.class_id.copy()
        if incoming_detections.class_id is not None
        else None
    )
    BYTE_TRACKER.reset()
    with exception:
        for i in range(3):
            incoming_detections.xyxy += 1
            if with_mask:
                mask = np.zeros(
                    (incoming_detections.xyxy.shape[0], IMG_HEIGHT, IMG_WIDTH)
                )
                for i in range(incoming_detections.xyxy.shape[0]):
                    mask[
                        i,  # box
                        int(incoming_detections.xyxy[i, 1]) : int(
                            incoming_detections.xyxy[i, 3]
                        ),  # y
                        int(incoming_detections.xyxy[i, 0]) : int(
                            incoming_detections.xyxy[i, 2]
                        ),  # x
                    ] = 1
                incoming_detections.mask = mask
            tracked_detections = BYTE_TRACKER.update_with_detections(
                incoming_detections
            )

        assert tracked_detections == target_detections


TARGET_STRACKS = [
    STrack(
        tlwh=[det[0], det[1], det[2] - det[0], det[3] - det[1]],
        score=det[4],
        class_ids=det[5],
    )
    for det in TARGET_PREDICTIONS
]

for i, track in enumerate(TARGET_STRACKS):
    track.start_frame = 1
    track.frame_id = 3
    track.track_id = i + 1


@pytest.mark.parametrize(
    "tensors, target_stracks," " exception",
    [
        (  # Test baseline detection tensors with some high confidence values
            PREDICTIONS.copy(),
            TARGET_STRACKS.copy(),
            DoesNotRaise(),
        ),
        (  # Test baseline detection tensors with all low confidence values
            LOW_CONFIDENCE_PREDICTIONS,
            [],
            DoesNotRaise(),
        ),
    ],
)
def test_update_with_tensors(
    tensors: np.ndarray, target_stracks: List[STrack], exception: Exception
):
    BYTE_TRACKER.reset()
    with exception:
        for i in range(3):
            tensors[:, :4] += 1
            stracks = BYTE_TRACKER.update_with_tensors(tensors[:, :6])

        assert_strack_lists_equal(stracks, target_stracks)

TARGET_STRACKS = [
    STrack(
        tlwh=[det[0], det[1], det[2] - det[0], det[3] - det[1]],
        score=det[4],
        class_ids=det[5],
    )
    for det in TARGET_PREDICTIONS
]

# different end frame
for i, track in enumerate(TARGET_STRACKS):
    track.start_frame = 1
    track.frame_id = 1
    track.track_id = i + 1

@pytest.mark.parametrize(
    "tensors, target_stracks,"
    " exception",
    [
        (
            PREDICTIONS,
            TARGET_STRACKS,
            DoesNotRaise(),
        )
    ],
)
def test_tracker_reset(
    tensors: np.ndarray,
    target_stracks: List[STrack],
    exception: Exception
    ):
    byte_tracker = ByteTrack()
    with exception:
        for i in range(3):
            stracks = byte_tracker.update_with_tensors(tensors)
            tensors[:, :4] += 1

        byte_tracker.reset()
        stracks = byte_tracker.update_with_tensors(tensors=tensors)
        assert_strack_lists_equal(stracks, target_stracks)
