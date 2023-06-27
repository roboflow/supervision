from typing import List, Optional, Tuple
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, parse, tostring

import numpy as np

from supervision.dataset.utils import approximate_mask_with_polygons
from supervision.detection.core import Detections
from supervision.detection.utils import polygon_to_xyxy


def object_to_pascal_voc(
    xyxy: np.ndarray, name: str, polygon: Optional[np.ndarray] = None
) -> Element:
    root = Element("object")

    object_name = SubElement(root, "name")
    object_name.text = name

    bndbox = SubElement(root, "bndbox")
    xmin = SubElement(bndbox, "xmin")
    xmin.text = str(int(xyxy[0]))
    ymin = SubElement(bndbox, "ymin")
    ymin.text = str(int(xyxy[1]))
    xmax = SubElement(bndbox, "xmax")
    xmax.text = str(int(xyxy[2]))
    ymax = SubElement(bndbox, "ymax")
    ymax.text = str(int(xyxy[3]))

    if polygon is not None:
        object_polygon = SubElement(root, "polygon")
        for index, point in enumerate(polygon, start=1):
            x_coordinate, y_coordinate = point
            x = SubElement(object_polygon, f"x{index}")
            x.text = str(x_coordinate)
            y = SubElement(object_polygon, f"y{index}")
            y.text = str(y_coordinate)

    return root


def detections_to_pascal_voc(
    detections: Detections,
    classes: List[str],
    filename: str,
    image_shape: Tuple[int, int, int],
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
) -> str:
    """
    Converts Detections object to Pascal VOC XML format.

    Args:
        detections (Detections): A Detections object containing bounding boxes, class ids, and other relevant information.
        classes (List[str]): A list of class names corresponding to the class ids in the Detections object.
        filename (str): The name of the image file associated with the detections.
        image_shape (Tuple[int, int, int]): The shape of the image file associated with the detections.
        min_image_area_percentage (float): Minimum detection area relative to area of image associated with it.
        max_image_area_percentage (float): Maximum detection area relative to area of image associated with it.
        approximation_percentage (float): The percentage of polygon points to be removed from the input polygon, in the range [0, 1).
    Returns:
        str: An XML string in Pascal VOC format representing the detections.
    """
    height, width, depth = image_shape

    # Create root element
    annotation = Element("annotation")

    # Add folder element
    folder = SubElement(annotation, "folder")
    folder.text = "VOC"

    # Add filename element
    file_name = SubElement(annotation, "filename")
    file_name.text = filename

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
    for xyxy, mask, _, class_id, _ in detections:
        name = classes[class_id]
        if mask is not None:
            polygons = approximate_mask_with_polygons(
                mask=mask,
                min_image_area_percentage=min_image_area_percentage,
                max_image_area_percentage=max_image_area_percentage,
                approximation_percentage=approximation_percentage,
            )
            for polygon in polygons:
                xyxy = polygon_to_xyxy(polygon=polygon)
                next_object = object_to_pascal_voc(
                    xyxy=xyxy, name=name, polygon=polygon
                )
                annotation.append(next_object)
        else:
            next_object = object_to_pascal_voc(xyxy=xyxy, name=name)
            annotation.append(next_object)

    # Generate XML string
    xml_string = parseString(tostring(annotation)).toprettyxml(indent="  ")

    return xml_string


def load_pascal_voc_annotations(
    annotation_path: str,
) -> Tuple[str, Detections, List[str]]:
    """
    Loads PASCAL VOC XML annotations and returns the image name, a Detections instance, and a list of class names.

    Args:
        annotation_path (str): The path to the PASCAL VOC XML annotations file.

    Returns:
        Tuple[str, Detections, List[str]]: A tuple containing the image name, a Detections instance, and a list of class names of objects in the detections.
    """
    tree = parse(annotation_path)
    root = tree.getroot()

    image_name = root.find("filename").text

    xyxy = []
    class_names = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        class_names.append(class_name)

        bbox = obj.find("bndbox")
        x1 = int(bbox.find("xmin").text)
        y1 = int(bbox.find("ymin").text)
        x2 = int(bbox.find("xmax").text)
        y2 = int(bbox.find("ymax").text)

        xyxy.append([x1, y1, x2, y2])

    xyxy = np.array(xyxy)
    detections = Detections(xyxy=xyxy)

    return image_name, detections, class_names
