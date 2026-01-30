import warnings

warnings.warn(
    "The 'supervision.keypoint' module is deprecated in `0.27.0` and will be removed "
    "in `0.30.0`. Please use 'supervision.key_points' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from supervision.key_points.annotators import (  # noqa: F401, E402
    BaseKeyPointAnnotator,
    EdgeAnnotator,
    VertexAnnotator,
    VertexLabelAnnotator,
)
