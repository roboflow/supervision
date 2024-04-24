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


SKELETONS_BY_EDGE_COUNT: Dict[int, Edges] = {}
SKELETONS_BY_VERTEX_COUNT: Dict[int, Edges] = {}

for skeleton in Skeleton:
    SKELETONS_BY_EDGE_COUNT[len(skeleton.value)] = skeleton.value

    unique_vertices = set(vertex for edge in skeleton.value for vertex in edge)
    SKELETONS_BY_VERTEX_COUNT[len(unique_vertices)] = skeleton.value
