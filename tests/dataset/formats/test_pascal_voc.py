from __future__ import annotations

from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest
from defusedxml import ElementTree

from supervision.dataset.formats.pascal_voc import (
    detections_from_xml_obj,
    object_to_pascal_voc,
    parse_polygon_points,
)
from tests.helpers import _create_detections


def are_xml_elements_equal(elem1, elem2):
    if (
        elem1.tag != elem2.tag
        or elem1.attrib != elem2.attrib
        or elem1.text != elem2.text
        or len(elem1) != len(elem2)
    ):
        return False

    for child1, child2 in zip(elem1, elem2):
        if not are_xml_elements_equal(child1, child2):
            return False

    return True


@pytest.mark.parametrize(
    ("xyxy", "name", "polygon", "expected_result", "exception"),
    [
        pytest.param(
            np.array([0, 0, 10, 10]),
            "test",
            None,
            ElementTree.fromstring(
                """<object><name>test</name><bndbox><xmin>1</xmin><ymin>1</ymin>
                <xmax>11</xmax><ymax>11</ymax></bndbox></object>"""
            ),
            DoesNotRaise(),
            id="bbox_only",
        ),
        pytest.param(
            np.array([0, 0, 10, 10]),
            "test",
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
            ElementTree.fromstring(
                """<object><name>test</name><bndbox><xmin>1</xmin><ymin>1</ymin>
                <xmax>11</xmax><ymax>11</ymax>
                </bndbox><polygon><x1>1</x1><y1>1</y1><x2>11</x2>
                <y2>1</y2><x3>11</x3><y3>11</y3><x4>1</x4><y4>11</y4>
                </polygon></object>"""
            ),
            DoesNotRaise(),
            id="bbox_and_polygon",
        ),
    ],
)
def test_object_to_pascal_voc(
    xyxy: np.ndarray,
    name: str,
    polygon: np.ndarray | None,
    expected_result,
    exception: Exception,
):
    with exception:
        result = object_to_pascal_voc(xyxy=xyxy, name=name, polygon=polygon)
        assert are_xml_elements_equal(result, expected_result)


@pytest.mark.parametrize(
    ("polygon_element", "expected_result", "exception"),
    [
        pytest.param(
            ElementTree.fromstring(
                """<polygon><x1>0</x1><y1>0</y1><x2>10</x2><y2>0</y2><x3>10</x3>
                    <y3>10</y3><x4>0</x4><y4>10</y4></polygon>"""
            ),
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
            DoesNotRaise(),
            id="standard_polygon",
        )
    ],
)
def test_parse_polygon_points(
    polygon_element,
    expected_result: list[list],
    exception,
):
    with exception:
        result = parse_polygon_points(polygon_element)
        assert np.array_equal(result, expected_result)


ONE_CLASS_N_BBOX = """<annotation><object><name>test</name><bndbox><xmin>1</xmin>
<ymin>1</ymin><xmax>11</xmax><ymax>11</ymax>
</bndbox></object><object><name>test</name><bndbox><xmin>11</xmin><ymin>11</ymin>
<xmax>21</xmax><ymax>21</ymax></bndbox></object></annotation>"""


ONE_CLASS_ONE_BBOX = """<annotation><object><name>test</name><bndbox>
<xmin>1</xmin><ymin>1</ymin><xmax>11</xmax><ymax>11</ymax></bndbox></object>
</annotation>"""


N_CLASS_N_BBOX = """<annotation><object><name>test</name><bndbox><xmin>1</xmin>
<ymin>1</ymin><xmax>11</xmax><ymax>11</ymax>
</bndbox></object><object><name>test</name><bndbox>
<xmin>21</xmin><ymin>31</ymin><xmax>31</xmax><ymax>41</ymax></bndbox>
</object><object><name>test2</name><bndbox><xmin>
11</xmin><ymin>11</ymin><xmax>21</xmax><ymax>
21</ymax></bndbox></object></annotation>"""

NO_DETECTIONS = """<annotation></annotation>"""
MIXED_POLYGON_AND_BOX = """<annotation><object><name>test</name><bndbox>
<xmin>1</xmin><ymin>1</ymin><xmax>11</xmax><ymax>11</ymax></bndbox>
<polygon><x1>1</x1><y1>1</y1><x2>11</x2><y2>1</y2><x3>11</x3><y3>11</y3>
<x4>1</x4><y4>11</y4></polygon></object><object><name>test</name><bndbox>
<xmin>11</xmin><ymin>11</ymin><xmax>21</xmax><ymax>21</ymax></bndbox></object>
</annotation>"""


@pytest.mark.parametrize(
    (
        "xml_string",
        "classes",
        "resolution_wh",
        "force_masks",
        "expected_result",
        "exception",
    ),
    [
        pytest.param(
            ONE_CLASS_ONE_BBOX,
            ["test"],
            (100, 100),
            False,
            _create_detections(xyxy=[[0, 0, 10, 10]], class_id=[0]),
            DoesNotRaise(),
            id="one_class_one_bbox",
        ),
        pytest.param(
            ONE_CLASS_N_BBOX,
            ["test"],
            (100, 100),
            False,
            _create_detections(
                xyxy=np.array([[0, 0, 10, 10], [10, 10, 20, 20]]), class_id=[0, 0]
            ),
            DoesNotRaise(),
            id="one_class_n_bbox",
        ),
        pytest.param(
            N_CLASS_N_BBOX,
            ["test", "test2"],
            (100, 100),
            False,
            _create_detections(
                xyxy=np.array([[0, 0, 10, 10], [20, 30, 30, 40], [10, 10, 20, 20]]),
                class_id=[0, 0, 1],
            ),
            DoesNotRaise(),
            id="n_class_n_bbox",
        ),
        pytest.param(
            NO_DETECTIONS,
            [],
            (100, 100),
            False,
            _create_detections(xyxy=np.empty((0, 4)), class_id=[]),
            DoesNotRaise(),
            id="no_detections",
        ),
    ],
)
def test_detections_from_xml_obj(
    xml_string, classes, resolution_wh, force_masks, expected_result, exception
):
    with exception:
        root = ElementTree.fromstring(xml_string)
        result, _ = detections_from_xml_obj(root, classes, resolution_wh, force_masks)
        assert result == expected_result


@pytest.mark.parametrize("force_masks", [False, True])
def test_detections_from_xml_obj_mixed_polygon_and_bbox_masks_aligned(
    force_masks: bool,
) -> None:
    root = ElementTree.fromstring(MIXED_POLYGON_AND_BOX)
    detections, _ = detections_from_xml_obj(
        root=root,
        classes=["test"],
        resolution_wh=(30, 30),
        force_masks=force_masks,
    )

    assert detections.mask is not None
    assert detections.mask.shape == (2, 30, 30)
    assert detections.mask[0].any()
    assert not detections.mask[1].any()
