from enum import Enum
from typing import Optional, Tuple, List


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


def resolve_skeleton_by_vertex_count(count: int) -> Optional[List[Tuple[int, int]]]:
    for skeleton in Skeleton:
        unique_vertices = set(vertex for edge in skeleton.value for vertex in edge)
        if len(unique_vertices) == count:
            return skeleton.value
    return None


def resolve_skeleton_by_edge_count(count: int) -> Optional[List[Tuple[int, int]]]:
    for skeleton in Skeleton:
        if len(skeleton.value) == count:
            return skeleton.value
    return None
