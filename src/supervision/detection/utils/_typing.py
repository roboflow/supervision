from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

_DetectionDataValueType: TypeAlias = npt.NDArray[np.generic] | list[Any]
_DetectionDataType: TypeAlias = dict[str, _DetectionDataValueType]
_MetadataType: TypeAlias = dict[str, Any]
