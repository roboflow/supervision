from typing import Optional

import cv2
import numpy as np

from supervision.detection.utils import generate_2d_mask
from supervision.draw.color import Color
from supervision.geometry.core import Point, Rect


def draw_line(
    scene: np.ndarray, start: Point, end: Point, color: Color, thickness: int = 2
) -> np.ndarray:
    """
    Draws a line on a given scene.

    Parameters:
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


def draw_text(
    scene: np.ndarray,
    text: str,
    text_anchor: Point,
    text_color: Color = Color.black(),
    text_scale: float = 0.5,
    text_thickness: int = 1,
    text_padding: int = 10,
    text_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    background_color: Optional[Color] = None,
) -> np.ndarray:
    """
    Draw text on a scene.

    This function takes in a 2-dimensional numpy ndarray representing an image or scene, and draws text on it using OpenCV's putText function. The text is anchored at a specified Point, and its appearance can be customized using arguments such as color, scale, and font. An optional background color and padding can be specified to draw a rectangle behind the text.

    Parameters:
        scene (np.ndarray): A 2-dimensional numpy ndarray representing an image or scene.
        text (str): The text to be drawn.
        text_anchor (Point): The anchor point for the text, represented as a Point object with x and y attributes.
        text_color (Color, optional): The color of the text. Defaults to black.
        text_scale (float, optional): The scale of the text. Defaults to 0.5.
        text_thickness (int, optional): The thickness of the text. Defaults to 1.
        text_padding (int, optional): The amount of padding to add around the text when drawing a rectangle in the background. Defaults to 10.
        text_font (int, optional): The font to use for the text. Defaults to cv2.FONT_HERSHEY_SIMPLEX.
        background_color (Color, optional): The color of the background rectangle, if one is to be drawn. Defaults to None.

    Returns:
        np.ndarray: The input scene with the text drawn on it.

    Examples:
        ```python
        >>> scene = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> text_anchor = Point(x=50, y=50)
        >>> scene = draw_text(scene=scene, text="Hello, world!", text_anchor=text_anchor)
        ```
    """
    text_width, text_height = cv2.getTextSize(
        text=text,
        fontFace=text_font,
        fontScale=text_scale,
        thickness=text_thickness,
    )[0]
    text_rect = Rect(
        x=text_anchor.x - text_width // 2,
        y=text_anchor.y - text_height // 2,
        width=text_width,
        height=text_height,
    ).pad(text_padding)

    if background_color is not None:
        scene = draw_filled_rectangle(
            scene=scene, rect=text_rect, color=background_color
        )

    cv2.putText(
        img=scene,
        text=text,
        org=(text_anchor.x - text_width // 2,
             text_anchor.y + text_height // 2),
        fontFace=text_font,
        fontScale=text_scale,
        color=text_color.as_bgr(),
        thickness=text_thickness,
        lineType=cv2.LINE_AA,
    )
    return scene


def copy_paste(source_image: np.ndarray, source_polygon: np.ndarray, target_image: np.ndarray, scale: float = 1.0, x: int = 0, y: int = 0):
    """
    Copy and paste a region from a source image into a target image.

    Attributes:
        source_image (np.ndarray): The image to be copied.
        source_polygon (np.ndarray): The polygon area to be copied.
        target_image (np.ndarray): The image to be pasted into.
        scale: scale factor to apply to the source image
        x: x offset to paste into the target image at
        y: y offset to paste into the target image at


    Returns:
        np.ndarray: The target image with the source image pasted into it.
    """
    # cv2.imshow('source', source_image)
    # cv2.imshow('target', target_image)
    # # print(source_image.shape)

    output = target_image.copy()

    # generate mask image based on polygon
    source_width = source_image.shape[0]
    source_height = source_image.shape[1]
    mask = generate_2d_mask(source_polygon, (source_width, source_height))

    # cutout = cv2.bitwise_and(self.crop, self.crop, mask=self.mask)

    # scale the source and mask
    scaled_source = cv2.resize(source_image, None, fx=scale, fy=scale)
    scaled_mask = cv2.resize(mask, None, fx=scale, fy=scale)
    scaled_with = scaled_source.shape[0]
    scaled_height = scaled_source.shape[1]

    # generate a patch from the source image  / with background from target image where we are pasting
    patch = np.where(
        np.expand_dims(scaled_mask, axis=2),
        scaled_source,
        target_image[y:y+scaled_height, x:x+scaled_with, :]
    )

    # paste the patch area into the output image
    output[y:y+scaled_height, x:x+scaled_with, :] = patch

    return output
