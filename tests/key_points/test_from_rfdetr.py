import numpy as np
import pytest

import supervision as sv


def test_keypoints_from_rfdetr_detections() -> None:
    """Converts RF-DETR detections.data['keypoints'] into a KeyPoints object."""
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10], [10, 10, 20, 20]], dtype=np.float32),
        class_id=np.array([1, 3], dtype=int),
        data={
            "keypoints": np.array(
                [
                    [[1.0, 2.0, 0.9], [3.0, 4.0, 0.8]],
                    [[5.0, 6.0, 0.7], [7.0, 8.0, 0.6]],
                ],
                dtype=np.float32,
            )
        },
    )

    key_points = sv.KeyPoints.from_rfdetr(detections)

    assert key_points.xy.shape == (2, 2, 2)
    assert key_points.confidence is not None
    assert key_points.confidence.shape == (2, 2)
    assert key_points.class_id is not None
    assert np.array_equal(key_points.class_id, np.array([1, 3], dtype=int))


def test_keypoints_from_rfdetr_missing_keypoints_raises_clear_error() -> None:
    """Missing detections.data['keypoints'] raises a clear conversion error."""
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
    )

    with pytest.raises(ValueError, match=r"data\['keypoints'\]"):
        sv.KeyPoints.from_rfdetr(detections)


def test_keypoints_from_rfdetr_malformed_shape_raises_clear_error() -> None:
    """Malformed keypoints shape raises a clear conversion error."""
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
        data={"keypoints": np.array([[[1.0, 2.0]]], dtype=np.float32)},
    )

    with pytest.raises(ValueError, match="shape \\(N, K, 3\\)"):
        sv.KeyPoints.from_rfdetr(detections)


def test_keypoint_annotator_uses_vertex_and_edge_rendering() -> None:
    """Converted RF-DETR keypoints are consumable by vertex and edge annotators."""
    scene = np.zeros((32, 32, 3), dtype=np.uint8)
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
        data={
            "keypoints": np.array(
                [[[10.0, 10.0, 0.9], [20.0, 20.0, 0.8]]], dtype=np.float32
            )
        },
    )
    key_points = sv.KeyPoints.from_rfdetr(detections)

    scene = sv.VertexAnnotator().annotate(scene=scene, key_points=key_points)
    scene = sv.EdgeAnnotator(edges=[(1, 2)]).annotate(
        scene=scene, key_points=key_points
    )

    assert np.any(scene != 0)
