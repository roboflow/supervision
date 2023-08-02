import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, parse, tostring

import cv2
import numpy as np

from supervision.dataset.utils import approximate_mask_with_polygons
from supervision.detection.core import Detections
from supervision.detection.utils import polygon_to_mask, polygon_to_xyxy
from supervision.utils.file import list_files_with_extensions


class XMLBuilder:
    default_root_name = "annotation"
    default_folder_name = "VOC"

    def __init__(
        self,
        filename: str,
        root: str = default_root_name,
        folder: str = default_folder_name,
    ):
        self.annotation = self.add_annot_root(
            root_name=root, folder=folder, filename=filename
        )

    def add_annot_root(
        self,
        filename: str,
        root_name: str = default_root_name,
        folder: str = default_folder_name,
    ) -> Element:
        annotation = Element(root_name)

        source = SubElement(annotation, "source")
        database = SubElement(source, "database")
        database.text = "roboflow.ai"

        folder = SubElement(annotation, "folder")
        folder.text = folder

        file_name = SubElement(annotation, "filename")
        file_name.text = filename

        return annotation

    def add_size_spec(self, width: int, height: int, depth: int) -> Element:
        size = SubElement(self.annotation, "size")
        w = SubElement(size, "width")
        w.text = str(width)
        h = SubElement(size, "height")
        h.text = str(height)
        d = SubElement(size, "depth")
        d.text = str(depth)
        return size

    def add_segmented(self, segmented_name: int = 0) -> Element:
        segmented = SubElement(self.annotation, "segmented")
        segmented.text = str(segmented_name)
        return segmented

    def add_object(self, name: str) -> Element:
        object = SubElement(self.annotation, "object")
        object_name = SubElement(object, "name")
        object_name.text = name
        return object

    def to_string(self) -> str:
        return parseString(tostring(self.annotation)).toprettyxml(indent="  ")

    def add_obj_det(self, root: Element, xyxy: np.ndarray) -> Element:
        bndbox = SubElement(root, "bndbox")
        xmin = SubElement(bndbox, "xmin")
        xmin.text = str(int(xyxy[0]))
        ymin = SubElement(bndbox, "ymin")
        ymin.text = str(int(xyxy[1]))
        xmax = SubElement(bndbox, "xmax")
        xmax.text = str(int(xyxy[2]))
        ymax = SubElement(bndbox, "ymax")
        ymax.text = str(int(xyxy[3]))

        return bndbox

    def add_img_segm(
        self,
        root: Element,
        polygon: np.ndarray,
    ) -> Element:
        object_polygon = SubElement(root, "polygon")
        for index, point in enumerate(polygon, start=1):
            x_coordinate, y_coordinate = point
            x = SubElement(object_polygon, f"x{index}")
            x.text = str(x_coordinate)
            y = SubElement(object_polygon, f"y{index}")
            y.text = str(y_coordinate)

        return object_polygon


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
        detections (Detections): A Detections object containing class ids and either bounding boxes or masks.
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

    xml_builder = XMLBuilder(filename)

    xml_builder.add_size_spec(width, height, depth)

    xyxy_and_mask_available = (
        detections.xyxy is not None and detections.mask is not None
    )
    mask_available = detections.mask is not None
    xyxy_available = detections.xyxy is not None

    if xyxy_and_mask_available or mask_available:
        if xyxy_and_mask_available:
            warnings.warn(
                "Detections object contains both xyxy and mask information. This will be removed in future versions. Please use either xyxy or mask."
            )
        xml_builder.add_segmented()
        for mask, class_id in zip(detections.mask, detections.class_id):
            name = classes[class_id]
            polygons = approximate_mask_with_polygons(
                mask=mask,
                min_image_area_percentage=min_image_area_percentage,
                max_image_area_percentage=max_image_area_percentage,
                approximation_percentage=approximation_percentage,
            )
            for polygon in polygons:
                next_object = xml_builder.add_object(name)
                if xyxy_available:
                    xyxy = polygon_to_xyxy(polygon=polygon)
                    xml_builder.add_obj_det(root=next_object, xyxy=xyxy)
                xml_builder.add_img_segm(root=next_object, polygon=polygon)

    elif xyxy_available:
        for xyxy, class_id in zip(detections.xyxy, detections.class_id):
            name = classes[class_id]
            next_object = xml_builder.add_object(name)
            xml_builder.add_obj_det(root=next_object, xyxy=xyxy)

    xml_string = xml_builder.to_string()

    return xml_string


def load_pascal_voc_annotations(
    images_directory_path: str,
    annotations_directory_path: str,
    force_masks: bool = False,
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, Detections]]:
    """
    Loads PASCAL VOC annotations and returns class names, images, and their corresponding detections.

    Args:
        images_directory_path (str): The path to the directory containing the images.
        annotations_directory_path (str): The path to the directory containing the PASCAL VOC annotation files.
        force_masks (bool, optional): If True, forces masks to be loaded for all annotations, regardless of whether they are present.

    Returns:
        Tuple[List[str], Dict[str, np.ndarray], Dict[str, Detections]]: A tuple containing a list of class names, a dictionary with image names as keys and images as values, and a dictionary with image names as keys and corresponding Detections instances as values.
    """

    image_paths = list_files_with_extensions(
        directory=images_directory_path, extensions=["jpg", "jpeg", "png"]
    )

    classes = []
    images = {}
    annotations = {}

    for image_path in image_paths:
        image_name = Path(image_path).stem
        image = cv2.imread(str(image_path))

        annotation_path = os.path.join(annotations_directory_path, f"{image_name}.xml")
        if not os.path.exists(annotation_path):
            images[image_path.name] = image
            annotations[image_path.name] = Detections.empty()
            continue

        tree = parse(annotation_path)
        root = tree.getroot()

        resolution_wh = (image.shape[1], image.shape[0])
        annotation, classes = detections_from_xml_obj(
            root, classes, resolution_wh, force_masks
        )

        images[image_path.name] = image
        annotations[image_path.name] = annotation

    return classes, images, annotations


def detections_from_xml_obj(
    root: Element, classes: List[str], resolution_wh, force_masks: bool = False
) -> Tuple[Detections, List[str]]:
    """
    Converts an XML object in Pascal VOC format to a Detections object.
    Expected XML format:
    <annotation>
        ...
        <object>
            <name>dog</name>
            <bndbox>
                <xmin>48</xmin>
                <ymin>240</ymin>
                <xmax>195</xmax>
                <ymax>371</ymax>
            </bndbox>
            <polygon>
                <x1>48</x1>
                <y1>240</y1>
                <x2>195</x2>
                <y2>240</y2>
                <x3>195</x3>
                <y3>371</y3>
                <x4>48</x4>
                <y4>371</y4>
            </polygon>
        </object>
    </annotation>

    Returns:
        Tuple[Detections, List[str]]: A tuple containing a Detections object and an updated list of class names, extended with the class names from the XML object.
    """
    xyxy = []
    class_names = []
    masks = []
    with_masks = False
    extended_classes = classes[:]
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        class_names.append(class_name)

        bbox = obj.find("bndbox")
        x1 = int(bbox.find("xmin").text)
        y1 = int(bbox.find("ymin").text)
        x2 = int(bbox.find("xmax").text)
        y2 = int(bbox.find("ymax").text)

        xyxy.append([x1, y1, x2, y2])

        with_masks = obj.find("polygon") is not None
        with_masks = force_masks if force_masks else with_masks

        for polygon in obj.findall("polygon"):
            polygon_points = parse_polygon_points(polygon)

            mask_from_polygon = polygon_to_mask(
                polygon=np.array(polygon_points),
                resolution_wh=resolution_wh,
            )
            masks.append(mask_from_polygon)

    xyxy = np.array(xyxy) if len(xyxy) > 0 else np.empty((0, 4))
    for k in set(class_names):
        if k not in extended_classes:
            extended_classes.append(k)
    class_id = np.array(
        [extended_classes.index(class_name) for class_name in class_names]
    )

    if with_masks:
        annotation = Detections(
            xyxy=xyxy, mask=np.array(masks).astype(bool), class_id=class_id
        )
    else:
        annotation = Detections(xyxy=xyxy, class_id=class_id)
    return annotation, extended_classes


def parse_polygon_points(polygon: Element) -> List[List[int]]:
    polygon_points = []
    coords = polygon.findall(".//*")
    for i in range(0, len(coords), 2):
        x = int(coords[i].text)
        y = int(coords[i + 1].text)
        polygon_points.append([x, y])
    return polygon_points
