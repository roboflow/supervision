from typing import List, Tuple
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring

from supervision.detection.core import Detections


def detections_to_pascal_voc(
        detections: Detections,
        classes: List[str],
        filename: str,
        image_shape: Tuple[int, int, int]
) -> str:
    """
    Converts Detections object to Pascal VOC XML format.

    Args:
        detections (Detections): A Detections object containing bounding boxes, class ids, and other relevant information.
        classes (List[str]): A list of class names corresponding to the class ids in the Detections object.
        filename (str): The name of the image file associated with the detections.
        image_shape (Tuple[int, int, int]): The shape of the image file associated with the detections.
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


# def dataset_to_pascal_voc(
#         dataset: Dataset,
#         images_directory_path: Optional[str],
#         annotations_directory_path: Optional[str]
# ) -> None:
#     # Ensure the output directories exist
#     if images_directory_path and not os.path.exists(images_directory_path):
#         os.makedirs(images_directory_path)
#
#     if annotations_directory_path and not os.path.exists(annotations_directory_path):
#         os.makedirs(annotations_directory_path)
#
#     # Iterate over the images and their corresponding detections
#     for image_path, image_annotations in dataset.annotations.items():
#         # Retrieve the image from the dataset
#         image = dataset.images[image_path]
#
#         # Create the root element for the XML
#         annotation = ET.Element("annotation")
#
#         # Add folder, filename, and path elements
#         folder = ET.SubElement(annotation, "folder")
#         folder.text = ""
#         filename = ET.SubElement(annotation, "filename")
#         filename.text = os.path.basename(image_path)
#         path = ET.SubElement(annotation, "path")
#         path.text = os.path.basename(image_path)
#
#         # Add image size information using the image shape
#         size = ET.SubElement(annotation, "size")
#         width = ET.SubElement(size, "width")
#         width.text = str(image.shape[1])  # Use image width
#         height = ET.SubElement(size, "height")
#         height.text = str(image.shape[0])  # Use image height
#         depth = ET.SubElement(size, "depth")
#         depth.text = str(image.shape[2])  # Use image depth (number of channels)
#
#         # Iterate over the bounding boxes and their attributes
#         for i in range(image_annotations.xyxy.shape[0]):
#             object_element = ET.SubElement(annotation, "object")
#
#             name = ET.SubElement(object_element, "name")
#             name.text = dataset.classes[image_annotations.class_id[i]]
#
#             pose = ET.SubElement(object_element, "pose")
#             pose.text = "Unspecified"
#
#             truncated = ET.SubElement(object_element, "truncated")
#             truncated.text = "0"
#
#             difficult = ET.SubElement(object_element, "difficult")
#             difficult.text = "0"
#
#             # Add bounding box information
#             bndbox = ET.SubElement(object_element, "bndbox")
#             xmin = ET.SubElement(bndbox, "xmin")
#             xmin.text = str(image_annotations.xyxy[i, 0])
#             ymin = ET.SubElement(bndbox, "ymin")
#             ymin.text = str(image_annotations.xyxy[i, 1])
#             xmax = ET.SubElement(bndbox, "xmax")
#             xmax.text = str(image_annotations.xyxy[i, 2])
#             ymax = ET.SubElement(bndbox, "ymax")
#             ymax.text = str(image_annotations.xyxy[i, 3])
#
#         # Save the XML to the annotations directory
#         if annotations_directory_path:
#             xml_file_path = os.path.join(annotations_directory_path,
#                                          os.path.splitext(os.path.basename(image_path))[0] + ".xml")
#             tree = ET.ElementTree(annotation)
#             tree.write(xml_file_path, encoding="utf-8", xml_declaration=True)
#
#         # Save the image to the images directory
#         if images_directory_path:
#             img_file_path = os.path.join(images_directory_path, os.path.basename(image_path))
#             cv2.imwrite(img_file_path, dataset.images[image_path])