

from typing import List, Tuple


class Skeleton:
    """
    The `Skeleton` class connects keypoints to form a skeleton.

    Args:
        limbs: (List[Tuple[int, int]]): List of (`class_id_1`,
        `class_id_2`) representing connected points of the skeleton.
    """

    limbs: List[Tuple[int, int]]

    def __init__(self, limbs: List[Tuple[int, int]]):
        self.limbs = limbs


NO_SKELETON = Skeleton([])
YOLO_V8_SKELETON = Skeleton(
    [
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
)


def _make_sequential_skeleton(limb_count: int) -> Skeleton:
    """
    Create a skeleton where neighboring keypoints are connected sequentially.

    Args:
        limb_count (int): The number of limbs to connect.

    Returns:
        Skeleton: The skeleton connecting all keypoints.
    """
    limbs = []
    for i in range(0, limb_count - 1):
        limbs.append((i, i + 1))
    limbs.append((0, limb_count - 1))
    return Skeleton(limbs)


class KnownSkeletons:
    """
    Helps automatically determine which skeleton to use.

    The `KnownSkeletons` class contains a collection of predefined skeletons
    for various keypoint models, to be distinguished by limb counts.

    Example:
        # Suppose you have a model returning 15 keypoints.
        skeleton = KnownSkeletons().get_skeleton(15)
    """
    _instance = None
    _skeletons: dict[int, Skeleton] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KnownSkeletons, cls).__new__(cls)
        return cls._instance

    def add_skeleton(self, skeleton: Skeleton) -> None:
        len_limbs = len(skeleton.limbs)
        if len_limbs in self._skeletons:
            raise ValueError(
                f"A skeleton with {len_limbs} limbs already exists")
        self._skeletons[len_limbs] = skeleton

    def get_skeleton(self, limb_count: int) -> Skeleton:
        if limb_count not in self._skeletons:
            print(
                f"Warning: No skeleton found with {limb_count} limbs. Will create one now.")
            self.add_skeleton(_make_sequential_skeleton(limb_count))
        return self._skeletons[limb_count]


KnownSkeletons().add_skeleton(NO_SKELETON)
KnownSkeletons().add_skeleton(YOLO_V8_SKELETON)
