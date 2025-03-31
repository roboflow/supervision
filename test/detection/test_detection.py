import unittest

import numpy as np

from supervision.detections.core import Detections


class TestDetectionsTransform(unittest.TestCase):
    def test_transform(self):
        # Mock dataset
        class DatasetMock:
            def __init__(self):
                self.classes = ["animal", "bird"]

        # Example detections
        detections = Detections(
            xyxy=np.array([[10, 10, 50, 50], [60, 60, 100, 100]]),
            confidence=np.array([0.9, 0.8]),
            class_id=np.array([0, 1]),
            data={"class_name": ["dog", "eagle"]},
        )

        # Class mapping
        class_mapping = {"dog": "animal", "eagle": "bird"}

        # Transform detections
        transformed_detections = detections.transform(DatasetMock(), class_mapping)

        # Verify results
        self.assertEqual(transformed_detections.class_id.tolist(), [0, 1])
        self.assertEqual(
            transformed_detections.data["class_name"].tolist(), ["animal", "bird"]
        )


if __name__ == "__main__":
    unittest.main()
