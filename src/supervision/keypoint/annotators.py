from supervision.utils.internal import warn_deprecated

warn_deprecated(
    "The 'supervision.keypoint' module is deprecated in `0.27.0` and will be removed "
    "in `0.30.0`. Please use 'supervision.key_points' instead."
)

from supervision.key_points.annotators import (  # noqa: E402, F401
    BaseKeyPointAnnotator,
    EdgeAnnotator,
    VertexAnnotator,
    VertexLabelAnnotator,
)
