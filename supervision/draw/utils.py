import os
from typing import Optional, Tuple, Union

import cv2
import numpy as np

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

    Parameters:
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
    Draws a filled rectangle on an image.

    Parameters:
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

    Parameters:
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
    text_color: Color = Color.BLACK,
    text_scale: float = 0.5,
    text_thickness: int = 1,
    text_padding: int = 10,
    text_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    background_color: Optional[Color] = None,
) -> np.ndarray:
    """
    Draw text with background on a scene.

    Parameters:
        scene (np.ndarray): A 2-dimensional numpy ndarray representing an image or scene
        text (str): The text to be drawn.
        text_anchor (Point): The anchor point for the text, represented as a
            Point object with x and y attributes.
        text_color (Color, optional): The color of the text. Defaults to black.
        text_scale (float, optional): The scale of the text. Defaults to 0.5.
        text_thickness (int, optional): The thickness of the text. Defaults to 1.
        text_padding (int, optional): The amount of padding to add around the text
            when drawing a rectangle in the background. Defaults to 10.
        text_font (int, optional): The font to use for the text.
            Defaults to cv2.FONT_HERSHEY_SIMPLEX.
        background_color (Color, optional): The color of the background rectangle,
            if one is to be drawn. Defaults to None.

    Returns:
        np.ndarray: The input scene with the text drawn on it.

    Examples:
        ```python
        import numpy as np

        scene = np.zeros((100, 100, 3), dtype=np.uint8)
        text_anchor = Point(x=50, y=50)
        scene = draw_text(scene=scene, text="Hello, world!",text_anchor=text_anchor)
        ```
    """
    text_width, text_height = cv2.getTextSize(
        text=text,
        fontFace=text_font,
        fontScale=text_scale,
        thickness=text_thickness,
    )[0]

    text_anchor_x, text_anchor_y = text_anchor.as_xy_int_tuple()

    text_rect = Rect(
        x=text_anchor_x - text_width // 2,
        y=text_anchor_y - text_height // 2,
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
        org=(text_anchor_x - text_width // 2, text_anchor_y + text_height // 2),
        fontFace=text_font,
        fontScale=text_scale,
        color=text_color.as_bgr(),
        thickness=text_thickness,
        lineType=cv2.LINE_AA,
    )
    return scene


def draw_image(
    scene: np.ndarray, image: Union[str, np.ndarray], opacity: float, rect: Rect
) -> np.ndarray:
    """
    Draws an image onto a given scene with specified opacity and dimensions.

    Args:
        scene (np.ndarray): Background image where the new image will be drawn.
        image (Union[str, np.ndarray]): Image to draw.
        opacity (float): Opacity of the image to be drawn.
        rect (Rect): Rectangle specifying where to draw the image.

    Returns:
        np.ndarray: The updated scene.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: For invalid opacity or rectangle dimensions.
    """

    # Validate and load image
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image path ('{image}') does not exist.")
        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)

    # Validate opacity
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("Opacity must be between 0.0 and 1.0.")

    # Validate rectangle dimensions
    if (
        rect.x < 0
        or rect.y < 0
        or rect.x + rect.width > scene.shape[1]
        or rect.y + rect.height > scene.shape[0]
    ):
        raise ValueError("Invalid rectangle dimensions.")

    # Resize and isolate alpha channel
    image = cv2.resize(image, (rect.width, rect.height))
    alpha_channel = (
        image[:, :, 3]
        if image.shape[2] == 4
        else np.ones((rect.height, rect.width), dtype=image.dtype) * 255
    )
    alpha_scaled = cv2.convertScaleAbs(alpha_channel * opacity)

    # Perform blending
    scene_roi = scene[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width]
    alpha_float = alpha_scaled.astype(np.float32) / 255.0
    blended_roi = cv2.convertScaleAbs(
        (1 - alpha_float[..., np.newaxis]) * scene_roi
        + alpha_float[..., np.newaxis] * image[:, :, :3]
    )

    # Update the scene
    scene[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width] = blended_roi

    return scene


def calculate_dynamic_text_scale(resolution_wh: Tuple[int, int]) -> float:
    """
    Calculate a dynamic font scale based on the resolution of an image.

    Parameters:
         resolution_wh (Tuple[int, int]): A tuple representing the width and height
                 of the image.

    Returns:
         float: The calculated font scale factor.
    """
    return min(resolution_wh) * 1e-3


def calculate_dynamic_line_thickness(resolution_wh: Tuple[int, int]) -> int:
    """
    Calculate a dynamic line thickness based on the resolution of an image.

    Parameters:
        resolution_wh (Tuple[int, int]): A tuple representing the width and height
                of the image.

    Returns:
        int: The calculated line thickness in pixels.
    """
    min_dimension = min(resolution_wh)
    if min_dimension < 480:
        return 2
    if min_dimension < 720:
        return 2
    if min_dimension < 1080:
        return 2
    if min_dimension < 2160:
        return 4
    else:
        return 4
