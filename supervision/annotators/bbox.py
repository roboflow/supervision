from typing import List

import numpy as np

from supervision.commons.dataclasses import Detection


class BoxAnnotator:

    def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        pass
