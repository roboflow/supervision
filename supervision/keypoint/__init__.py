from supervision.utils.deprecation import warn_deprecated

warn_deprecated(
    "The 'supervision.keypoint' module is deprecated in `0.27.0` and will be removed "
    "in `0.30.0`. Please use 'supervision.key_points' instead.",
    stacklevel=2,
)

from supervision.key_points.annotators import (
    EdgeAnnotator,
    VertexAnnotator,
    VertexLabelAnnotator,
)
from supervision.key_points.core import KeyPoints
