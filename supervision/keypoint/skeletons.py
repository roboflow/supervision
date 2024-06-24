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

    GHUM = [
        (1, 2),
        (1, 5),
        (2, 3),
        (3, 4),
        (4, 8),
        (5, 6),
        (6, 7),
        (7, 9),
        (10, 11),
        (12, 13),
        (12, 14),
        (12, 24),
        (13, 15),
        (13, 25),
        (14, 16),
        (15, 17),
        (16, 18),
        (15, 19),
        (16, 22),
        (17, 19),
        (17, 21),
        (17, 23),
        (18, 20),
        (19, 21),
        (24, 25),
        (24, 26),
        (25, 27),
        (26, 28),
        (27, 29),
        (28, 30),
        (28, 32),
        (29, 31),
        (29, 33),
        (30, 32),
        (31, 33),
    ]


SKELETONS_BY_EDGE_COUNT: Dict[int, Edges] = {}
SKELETONS_BY_VERTEX_COUNT: Dict[int, Edges] = {}

for skeleton in Skeleton:
    SKELETONS_BY_EDGE_COUNT[len(skeleton.value)] = skeleton.value

    unique_vertices = set(vertex for edge in skeleton.value for vertex in edge)
    SKELETONS_BY_VERTEX_COUNT[len(unique_vertices)] = skeleton.value
