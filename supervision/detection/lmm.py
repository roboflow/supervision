import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class LMM(Enum):
    PALIGEMMA = "paligemma"


REQUIRED_ARGUMENTS: Dict[LMM, List[str]] = {LMM.PALIGEMMA: ["resolution_wh"]}

ALLOWED_ARGUMENTS: Dict[LMM, List[str]] = {LMM.PALIGEMMA: ["resolution_wh", "classes"]}


def validate_lmm_and_kwargs(lmm: Union[LMM, str], kwargs: Dict[str, Any]) -> LMM:
    if isinstance(lmm, str):
        try:
            lmm = LMM(lmm.lower())
        except ValueError:
            raise ValueError(
                f"Invalid lmm value: {lmm}. Must be one of {[e.value for e in LMM]}"
            )

    required_args = REQUIRED_ARGUMENTS.get(lmm, [])
    for arg in required_args:
        if arg not in kwargs:
            raise ValueError(f"Missing required argument: {arg}")

    allowed_args = ALLOWED_ARGUMENTS.get(lmm, [])
    for arg in kwargs:
        if arg not in allowed_args:
            raise ValueError(f"Argument {arg} is not allowed for {lmm.name}")

    return lmm


def from_paligemma(
    result: str, resolution_wh: Tuple[int, int], classes: Optional[List[str]] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    w, h = resolution_wh
    pattern = re.compile(
        r"(?<!<loc\d{4}>)<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})> ([\w\s\-]+)"
    )
    matches = pattern.findall(result)
    matches = np.array(matches) if matches else np.empty((0, 5))

    xyxy, class_name = matches[:, [1, 0, 3, 2]], matches[:, 4]
    xyxy = xyxy.astype(int) / 1024 * np.array([w, h, w, h])
    class_name = np.char.strip(class_name.astype(str))
    class_id = None

    if classes is not None:
        mask = np.array([name in classes for name in class_name]).astype(bool)
        xyxy, class_name = xyxy[mask], class_name[mask]
        class_id = np.array([classes.index(name) for name in class_name])

    return xyxy, class_id, class_name
