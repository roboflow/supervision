import cv2
import numpy as np

from supervision.commons.dataclasses import Point, Rect
from supervision.draw.color import Color


def draw_line(
    scene: np.ndarray, start: Point, end: Point, color: Color, thickness: int = 2
) -> np.ndarray:
    """
    Draws a line on a given scene.

    :param scene: np.ndarray : The scene on which the line will be drawn
    :param start: Point : The starting point of the line
    :param end: Point : The end point of the line
    :param color: Color : The color of the line
    :param thickness: int : The thickness of the line
    :return: np.ndarray : The scene with the line drawn on it
    """
    cv2.line(
        scene,
        start.as_xy_int_tuple(),
        end.as_xy_int_tuple(),
        color.as_bgr(),
        thickness=thickness,
    )
    return scene


def draw_rectangle(
    scene: np.ndarray, rect: Rect, color: Color, thickness: int = 2
) -> np.ndarray:
    """
    Draws a rectangle on an image.

    :param scene: np.ndarray : The image on which to draw the rectangle.
    :param rect: Rect : The rectangle to draw.
    :param color: Color : The color of the rectangle.
    :param thickness: int : The thickness of the rectangle border.
    :return: np.ndarray : The image with the rectangle drawn on it.
    """
    cv2.rectangle(
        scene,
        rect.top_left.as_xy_int_tuple(),
        rect.bottom_right.as_xy_int_tuple(),
        color.as_bgr(),
        thickness=thickness,
    )
    return scene


def draw_filled_rectangle(scene: np.ndarray, rect: Rect, color: Color) -> np.ndarray:
    """
    Draws a filled rectangle on the given scene.

    :param scene: np.ndarray : The scene on which to draw the rectangle.
    :param rect: Rect : The rectangle to be drawn.
    :param color: Color : The color of the rectangle.
    :return: np.ndarray : The updated scene with the filled rectangle drawn on it.
    """
    cv2.rectangle(
        scene,
        rect.top_left.as_xy_int_tuple(),
        rect.bottom_right.as_xy_int_tuple(),
        color.as_bgr(),
        -1,
    )
    return scene
