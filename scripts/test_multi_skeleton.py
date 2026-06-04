import numpy as np

import supervision as sv
from supervision.draw.color import Color

image = np.zeros((1000, 1000, 3), dtype=np.uint8)

xy = np.array(
    [
        # Instance 0: "person" (class_id=0) — 5 keypoints
        [[200, 100], [150, 300], [250, 300], [140, 550], [260, 550]],
        # Instance 1: "dog" (class_id=1) — 3 keypoints, padded to 5
        [[600, 200], [550, 400], [650, 400], [0, 0], [0, 0]],
        # Instance 2: another "person" (class_id=0)
        [[800, 100], [750, 300], [850, 300], [740, 550], [860, 550]],
    ],
    dtype=np.float32,
)

visible = np.array(
    [
        [True, True, True, True, True],
        [True, True, True, False, False],
        [True, True, True, True, True],
    ],
    dtype=bool,
)

key_points = sv.KeyPoints(
    xy=xy,
    class_id=np.array([0, 1, 0]),
    visible=visible,
)

PERSON_EDGES = [(1, 2), (1, 3), (2, 4), (3, 5)]
DOG_EDGES = [(1, 2), (1, 3)]

edge_annotator = sv.EdgeAnnotator(
    color=Color.GREEN,
    thickness=3,
    edges={0: PERSON_EDGES, 1: DOG_EDGES},
)

vertex_annotator = sv.VertexAnnotator(
    color=Color.RED,
    radius=8,
)

annotated = vertex_annotator.annotate(scene=image.copy(), key_points=key_points)
annotated = edge_annotator.annotate(scene=annotated, key_points=key_points)

import cv2

cv2.imwrite("multi_skeleton_test.png", annotated)
print("Saved to multi_skeleton_test.png")
