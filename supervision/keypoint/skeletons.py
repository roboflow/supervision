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


skeletons_by_edge_count: Dict[int, Edges] = {}
skeletons_by_vertex_count: Dict[int, Edges] = {}

for skeleton in Skeleton:
    skeletons_by_edge_count[len(skeleton.value)] = skeleton.value

    unique_vertices = set(
        vertex for edge in skeleton.value for vertex in edge)
    skeletons_by_vertex_count[len(unique_vertices)] = skeleton.value
