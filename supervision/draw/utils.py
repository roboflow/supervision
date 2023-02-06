import cv2
import numpy as np

from supervision.draw.color import Color
from supervision.geometry.dataclasses import Point, Rect


def draw_line(
    scene: np.ndarray, start: Point, end: Point, color: Color, thickness: int = 2
) -> np.ndarray:
    """
    Draws a line on a given scene.

    Attributes:

        scene (np.ndarray): The scene on which the line will be drawn
        start (Point): The starting point of the line
        end (Point): The end point of the line
        color (Color): The color of the line
        thickness (int): The thickness of the line

    Returns:
        np.ndarray: The scene with the line drawn on it
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

    Attributes:
        scene (np.ndarray): The scene on which the rectangle will be drawn
        rect (Rect): The rectangle to be drawn
        color (Color): The color of the rectangle
        thickness (int): The thickness of the rectangle border

    Returns:
        np.ndarray: The scene with the rectangle drawn on it

    Example:
        ```python
        >>> # TODO: Add example
        ```
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

    Attributes:
        scene (np.ndarray): The scene on which the rectangle will be drawn
        rect (Rect): The rectangle to be drawn
        color (Color): The color of the rectangle

    Returns:
        np.ndarray: The scene with the rectangle drawn on it

    Example:
        ```python
        >>> # TODO: Add example
        ```

    """
    cv2.rectangle(
        scene,
        rect.top_left.as_xy_int_tuple(),
        rect.bottom_right.as_xy_int_tuple(),
        color.as_bgr(),
        -1,
    )
    return scene


def draw_polygon(
    scene: np.ndarray, polygon: np.ndarray, color: Color, thickness: int = 2
) -> np.ndarray:
    """Draw a polygon on a scene.

    Attributes:
        scene (np.ndarray): The scene to draw the polygon on.
        polygon (np.ndarray): The polygon to be drawn, given as a list of vertices.
        color (Color): The color of the polygon.
        thickness (int, optional): The thickness of the polygon lines, by default 2.

    Returns:
        np.ndarray: The scene with the polygon drawn on it.
    """
    cv2.polylines(
        scene, [polygon], isClosed=True, color=color.as_bgr(), thickness=thickness
    )
    return scene
