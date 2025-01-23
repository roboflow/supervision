### `transform`

Transform detections to match the dataset's class names and IDs.

This method performs the following steps:

1. **Remaps class names** using the provided `class_mapping` dictionary.
2. **Filters out predictions** that are not present in the dataset's classes.
3. **Remaps class IDs** to match the dataset's class IDs.

#### Parameters

- **dataset**: The dataset object containing class names and IDs.
- **class_mapping** (`Optional[Dict[str, str]]`): A dictionary to map model class names to dataset class names. If `None`, no remapping is performed.

#### Returns

- **Detections**: A new `Detections` object with transformed class names and IDs.

#### Raises

- **ValueError**: If the dataset does not contain the required class names.

#### Example

```python
# Example dataset with class names
class DatasetMock:
    def __init__(self):
        self.classes = ["animal", "bird"]

# Example detections
detections = Detections(
    xyxy=np.array([[10, 10, 50, 50], [60, 60, 100, 100]]),
    confidence=np.array([0.9, 0.8]),
    class_id=np.array([0, 1]),
    data={"class_name": ["dog", "eagle"]}
)

# Class mapping
class_mapping = {"dog": "animal", "eagle": "bird"}

# Transform detections
transformed_detections = detections.transform(DatasetMock(), class_mapping)

print(transformed_detections.class_id)  # Output: [0, 1]
print(transformed_detections.data["class_name"])  # Output: ["animal", "bird"]
```
