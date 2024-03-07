import numpy as np
import pytest

from supervision import Detections, MatchPattern


@pytest.mark.parametrize(
    "constraints",
    [
        (
            [
                (1, ["Box1.class_id"]),
                (0.1, ["Box1.confidence"]),
                ([0, 0, 15, 15], ["Box2.xyxy"]),
            ]
        ),  # Test constraints with values
        (
            [
                (lambda id: id == 1, ["Box1.class_id"]),
                (lambda score: score == 0.1, ["Box1.confidence"]),
                (lambda xyxy: xyxy[3] == 15, ["Box2.xyxy"]),
            ]
        ),  # Test constraints with functions
        (
            [
                (lambda id: id == 1, ["Box1.class_id"]),
                (lambda xyxy1, xyxy2: xyxy1[0] == xyxy2[0], ["Box1.xyxy", "Box2.xyxy"]),
            ]
        ),  # Test constraints with multiple arguments
    ],
)
def test_match_pattern(constraints):
    detections = Detections(
        xyxy=np.array(
            [
                [0, 0, 10, 10],
                [0, 0, 15, 15],
                [5, 5, 20, 20],
            ]
        ),
        confidence=np.array([0.1, 0.2, 0.3]),
        class_id=np.array([1, 2, 3]),
    )

    expected_result = [
        Detections(
            xyxy=np.array(
                [
                    [0, 0, 10, 10],
                    [0, 0, 15, 15],
                ]
            ),
            confidence=np.array([0.1, 0.2]),
            class_id=np.array([1, 2]),
            data={"match_name": np.array(["Box1", "Box2"])},
        )
    ]

    matches = MatchPattern(constraints).match(detections)

    assert matches == expected_result


def test_match_pattern_with_2_results():
    detections = Detections(
        xyxy=np.array(
            [
                [0, 0, 10, 10],
                [0, 0, 15, 15],
                [5, 5, 20, 20],
            ]
        ),
        confidence=np.array([0.1, 0.2, 0.3]),
        class_id=np.array([1, 2, 3]),
    )

    expected_result = [
        Detections(
            xyxy=np.array(
                [
                    [0, 0, 10, 10],
                ]
            ),
            confidence=np.array([0.1]),
            class_id=np.array([1]),
            data={"match_name": np.array(["Box1"])},
        ),
        Detections(
            xyxy=np.array(
                [
                    [0, 0, 15, 15],
                ]
            ),
            confidence=np.array([0.2]),
            class_id=np.array([2]),
            data={"match_name": np.array(["Box1"])},
        ),
    ]

    matches = MatchPattern([[lambda xyxy: xyxy[0] == 0, ["Box1.xyxy"]]]).match(
        detections
    )

    assert matches == expected_result


def test_add_constraint():
    detections = Detections(
        xyxy=np.array(
            [
                [0, 0, 10, 10],
                [0, 0, 15, 15],
                [5, 5, 20, 20],
            ]
        ),
        confidence=np.array([0.1, 0.2, 0.3]),
        class_id=np.array([1, 2, 3]),
    )

    expected_result = [
        Detections(
            xyxy=np.array(
                [
                    [0, 0, 10, 10],
                ]
            ),
            confidence=np.array([0.1]),
            class_id=np.array([1]),
            data={"match_name": np.array(["Box1"])},
        )
    ]
    pattern = MatchPattern([])
    pattern.add_constraint(lambda id: id == 1, ["Box1.class_id"])
    matches = pattern.match(detections)
    assert matches == expected_result
