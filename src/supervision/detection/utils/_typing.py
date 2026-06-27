from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

DetectionDataValueType: TypeAlias = npt.NDArray[np.generic] | list[Any]
DetectionDataType: TypeAlias = dict[str, DetectionDataValueType]
MetadataType: TypeAlias = dict[str, Any]
