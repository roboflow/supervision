from typing import List
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring

from supervision.detection.core import Detections


def detections_to_voc_xml(
    detections: Detections,
    classes: List[str],
    filename: str,
    width: int,
    height: int,
    depth: int = 3,
) -> str:
    """
    Converts Detections object to Pascal VOC XML format.

    Args:
        detections (Detections): A Detections object containing bounding boxes, class ids, and other relevant information.
        classes (List[str]): A list of class names corresponding to the class ids in the Detections object.
        filename (str): The name of the image file associated with the detections.
        width (int): The width of the image in pixels.
        height (int): The height of the image in pixels.
        depth (int, optional): The number of color channels in the image. Defaults to 3 for RGB images.

    Returns:
        str: An XML string in Pascal VOC format representing the detections.

    Examples:
        ```python
        >>> import numpy as np
        >>> import supervision as sv

        >>> xyxy = np.array([
        ...     [50, 30, 200, 180],
        ...     [20, 40, 150, 190]
        ... ])
        >>> class_id = np.array([1, 0])
        >>> detections = Detections(xyxy=xyxy, class_id=class_id)

        >>> classes = ["dog", "cat"]

        >>> voc_xml = detections_to_voc_xml(
        ...     detections=detections,
        ...     classes=classes,
        ...     filename="image1.jpg",
        ...     width=500,
        ...     height=400
        ... )
        ```
    """

    # Create root element
    annotation = Element("annotation")

    # Add folder element
    folder = SubElement(annotation, "folder")
    folder.text = "VOC"

    # Add filename element
    fname = SubElement(annotation, "filename")
    fname.text = filename

    # Add source element
    source = SubElement(annotation, "source")
    database = SubElement(source, "database")
    database.text = "roboflow.ai"

    # Add size element
    size = SubElement(annotation, "size")
    w = SubElement(size, "width")
    w.text = str(width)
    h = SubElement(size, "height")
    h.text = str(height)
    d = SubElement(size, "depth")
    d.text = str(depth)

    # Add segmented element
    segmented = SubElement(annotation, "segmented")
    segmented.text = "0"

    # Add object elements
    for i in range(detections.xyxy.shape[0]):
        obj = SubElement(annotation, "object")

        class_id = detections.class_id[i] if detections.class_id is not None else None
        name = SubElement(obj, "name")
        name.text = classes[class_id] if class_id is not None else "unknown"

        bndbox = SubElement(obj, "bndbox")
        xmin = SubElement(bndbox, "xmin")
        xmin.text = str(int(detections.xyxy[i, 0]))
        ymin = SubElement(bndbox, "ymin")
        ymin.text = str(int(detections.xyxy[i, 1]))
        xmax = SubElement(bndbox, "xmax")
        xmax.text = str(int(detections.xyxy[i, 2]))
        ymax = SubElement(bndbox, "ymax")
        ymax.text = str(int(detections.xyxy[i, 3]))

    # Generate XML string
    xml_string = parseString(tostring(annotation)).toprettyxml(indent="  ")

    return xml_string
