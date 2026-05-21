from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

DetectionDataValueType: TypeAlias = npt.NDArray[np.generic] | list[object]
DetectionDataType: TypeAlias = dict[str, DetectionDataValueType]
MetadataType: TypeAlias = dict[str, object]
