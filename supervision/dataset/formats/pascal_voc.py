from typing import List, Optional, Tuple
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring

import cv2
import numpy as np

from supervision.detection.core import Detections
from supervision.detection.utils import (
    filter_polygons_by_area,
    mask_to_polygons,
    polygon_to_xyxy,
)


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
    minimum_detection_area_percentage: float = 0.0,
    maximum_detection_area_percentage: float = 1.0,
) -> str:
    """
    Converts Detections object to Pascal VOC XML format.

    Args:
        detections (Detections): A Detections object containing bounding boxes, class ids, and other relevant information.
        classes (List[str]): A list of class names corresponding to the class ids in the Detections object.
        filename (str): The name of the image file associated with the detections.
        image_shape (Tuple[int, int, int]): The shape of the image file associated with the detections.
        minimum_detection_area_percentage (float): Minimum detection area relative to area of image associated with it.
        maximum_detection_area_percentage (float): Maximum detection area relative to area of image associated with it.
    Returns:
        str: An XML string in Pascal VOC format representing the detections.
    """
    height, width, depth = image_shape
    image_area = height * width
    minimum_detection_area = minimum_detection_area_percentage * image_area
    maximum_detection_area = maximum_detection_area_percentage * image_area

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
            polygons = mask_to_polygons(mask=mask)
            if len(polygons) == 1:
                polygons = filter_polygons_by_area(
                    polygons=polygons, min_area=None, max_area=maximum_detection_area
                )
            else:
                polygons = filter_polygons_by_area(
                    polygons=polygons,
                    min_area=minimum_detection_area,
                    max_area=maximum_detection_area,
                )
            for polygon in polygons:
                approx_polygon = cv2.approxPolyDP(polygon, 1.0, True)
                approx_polygon = np.squeeze(approx_polygon, axis=1)
                xyxy = polygon_to_xyxy(polygon=approx_polygon)
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
