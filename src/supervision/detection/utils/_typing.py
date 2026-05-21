from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

_TypeDetectionDataValue: TypeAlias = npt.NDArray[np.generic] | list[object]
_TypeDetectionData: TypeAlias = dict[str, _TypeDetectionDataValue]
_TypeMetadata: TypeAlias = dict[str, object]
