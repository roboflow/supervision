from enum import Enum
from typing import Dict, List, Tuple

Edges = List[Tuple[int, int]]


class Skeleton(Enum):
    COCO = [
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (3, 5),
        (6, 12),
        (6, 7),
        (6, 8),
        (7, 13),
        (7, 9),
        (8, 10),
        (9, 11),
        (12, 13),
        (14, 12),
        (15, 13),
        (16, 14),
        (17, 15),
    ]

    # Hardcoding here, but it also comes within model results
    # Note: the classes in the model are 0-indexed, while the skeleton is 1-indexed
    YOLO_NAS = [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 6),
        (5, 7),
        (5, 11),
        (6, 8),
        (6, 12),
        (7, 9),
        (8, 10),
        (11, 12),
        (11, 13),
        (12, 14),
        (13, 15),
        (14, 16)
    ]


SKELETONS_BY_EDGE_COUNT: Dict[int, Edges] = {}
SKELETONS_BY_VERTEX_COUNT: Dict[int, Edges] = {}

for skeleton in Skeleton:
    SKELETONS_BY_EDGE_COUNT[len(skeleton.value)] = skeleton.value

    unique_vertices = set(vertex for edge in skeleton.value for vertex in edge)
    SKELETONS_BY_VERTEX_COUNT[len(unique_vertices)] = skeleton.value
