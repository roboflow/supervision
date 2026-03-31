from __future__ import annotations

import os
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from supervision.draw.color import Color
from supervision.geometry.core import Point, Rect


def draw_line(
    scene: npt.NDArray[np.uint8],
    start: Point,
    end: Point,
    color: Color = Color.ROBOFLOW,
    thickness: int = 2,
) -> npt.NDArray[np.uint8]:
    """
    Draws a line on a given scene.

    Args:
        scene: The scene on which the line will be drawn
        start: The starting point of the line
        end: The end point of the line
        color: The color of the line, defaults to Color.ROBOFLOW
        thickness: The thickness of the line

    Returns:
        The scene with the line drawn on it
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
    scene: npt.NDArray[np.uint8],
    rect: Rect,
    color: Color = Color.ROBOFLOW,
    thickness: int = 2,
) -> npt.NDArray[np.uint8]:
    """
    Draws a rectangle on an image.

    Args:
        scene: The scene on which the rectangle will be drawn
        rect: The rectangle to be drawn
        color: The color of the rectangle
        thickness: The thickness of the rectangle border

    Returns:
        The scene with the rectangle drawn on it
    """
    cv2.rectangle(
        scene,
        rect.top_left.as_xy_int_tuple(),
        rect.bottom_right.as_xy_int_tuple(),
        color.as_bgr(),
        thickness=thickness,
    )
    return scene


def draw_filled_rectangle(
    scene: npt.NDArray[np.uint8],
    rect: Rect,
    color: Color = Color.ROBOFLOW,
    opacity: float = 1,
) -> npt.NDArray[np.uint8]:
    """
    Draws a filled rectangle on an image.

    Args:
        scene: The scene on which the rectangle will be drawn
        rect: The rectangle to be drawn
        color: The color of the rectangle
        opacity: The opacity of rectangle when drawn on the scene.

    Returns:
        The scene with the rectangle drawn on it
    """
    if opacity == 1:
        cv2.rectangle(
            scene,
            rect.top_left.as_xy_int_tuple(),
            rect.bottom_right.as_xy_int_tuple(),
            color.as_bgr(),
            -1,
        )
    else:
        scene_with_annotations = scene.copy()
        cv2.rectangle(
            scene_with_annotations,
            rect.top_left.as_xy_int_tuple(),
            rect.bottom_right.as_xy_int_tuple(),
            color.as_bgr(),
            -1,
        )
        cv2.addWeighted(
            scene_with_annotations, opacity, scene, 1 - opacity, gamma=0, dst=scene
        )

    return scene


def draw_rounded_rectangle(
    scene: npt.NDArray[np.uint8],
    rect: Rect,
    color: Color,
    border_radius: int,
) -> npt.NDArray[np.uint8]:
    """
    Draws a rounded rectangle on an image.

    Args:
        scene: The image on which the rounded rectangle will be drawn.
        rect: The rectangle to be drawn.
        color: The color of the rounded rectangle.
        border_radius: The radius of the corner rounding.

    Returns:
        The image with the rounded rectangle drawn on it.
    """
    x1, y1, x2, y2 = rect.as_xyxy_int_tuple()
    width, height = x2 - x1, y2 - y1
    border_radius = min(border_radius, min(width, height) // 2)

    rectangle_coordinates = [
        ((x1 + border_radius, y1), (x2 - border_radius, y2)),
        ((x1, y1 + border_radius), (x2, y2 - border_radius)),
    ]
    circle_centers = [
        (x1 + border_radius, y1 + border_radius),
        (x2 - border_radius, y1 + border_radius),
        (x1 + border_radius, y2 - border_radius),
        (x2 - border_radius, y2 - border_radius),
    ]

    for coordinates in rectangle_coordinates:
        cv2.rectangle(
            img=scene,
            pt1=coordinates[0],
            pt2=coordinates[1],
            color=color.as_bgr(),
            thickness=-1,
        )
    for center in circle_centers:
        cv2.circle(
            img=scene,
            center=center,
            radius=border_radius,
            color=color.as_bgr(),
            thickness=-1,
        )
    return scene


def draw_polygon(
    scene: npt.NDArray[np.uint8],
    polygon: npt.NDArray[np.int_],
    color: Color = Color.ROBOFLOW,
    thickness: int = 2,
) -> npt.NDArray[np.uint8]:
    """Draw a polygon on a scene.

    Args:
        scene: The scene to draw the polygon on.
        polygon: The polygon to be drawn, given as a list of vertices.
        color: The color of the polygon. Defaults to Color.ROBOFLOW.
        thickness: The thickness of the polygon lines, by default 2.

    Returns:
        The scene with the polygon drawn on it.
    """
    cv2.polylines(
        scene, [polygon], isClosed=True, color=color.as_bgr(), thickness=thickness
    )
    return scene


def draw_filled_polygon(
    scene: npt.NDArray[np.uint8],
    polygon: npt.NDArray[np.int_],
    color: Color = Color.ROBOFLOW,
    opacity: float = 1,
) -> npt.NDArray[np.uint8]:
    """Draw a filled polygon on a scene.

    Args:
        scene: The scene to draw the polygon on.
        polygon: The polygon to be drawn, given as a list of vertices.
        color: The color of the polygon. Defaults to Color.ROBOFLOW.
        opacity: The opacity of polygon when drawn on the scene.

    Returns:
        The scene with the polygon drawn on it.
    """
    if opacity == 1:
        cv2.fillPoly(scene, [polygon], color=color.as_bgr())
    else:
        scene_with_annotations = scene.copy()
        cv2.fillPoly(scene_with_annotations, [polygon], color=color.as_bgr())
        cv2.addWeighted(
            scene_with_annotations, opacity, scene, 1 - opacity, gamma=0, dst=scene
        )

    return scene


def draw_text(
    scene: npt.NDArray[np.uint8],
    text: str,
    text_anchor: Point,
    text_color: Color = Color.BLACK,
    text_scale: float = 0.5,
    text_thickness: int = 1,
    text_padding: int = 10,
    text_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    background_color: Color | None = None,
) -> npt.NDArray[np.uint8]:
    """
    Draw text with background on a scene.

    Args:
        scene: A numpy ndarray representing the image, typically of shape
            (H, W, 3) for a color BGR image or (H, W) for grayscale,
            with dtype uint8.
        text: The text to be drawn.
        text_anchor: The anchor point for the text, represented as a
            Point object with x and y attributes.
        text_color: The color of the text. Defaults to black.
        text_scale: The scale of the text. Defaults to 0.5.
        text_thickness: The thickness of the text. Defaults to 1.
        text_padding: The amount of padding to add around the text
            when drawing a rectangle in the background. Defaults to 10.
        text_font: The font to use for the text.
            Defaults to cv2.FONT_HERSHEY_SIMPLEX.
        background_color: The color of the background rectangle,
            if one is to be drawn. Defaults to None.

    Returns:
        The input scene with the text drawn on it.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from supervision.geometry.core import Point
        >>> from supervision.draw.utils import draw_text
        >>> scene = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> text_anchor = Point(x=50, y=50)
        >>> scene = draw_text(
        ...     scene=scene, text="Hello, world!", text_anchor=text_anchor
        ... )
        >>> scene.shape
        (100, 100, 3)

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
    scene: npt.NDArray[np.uint8],
    image: str | npt.NDArray[np.uint8],
    opacity: float,
    rect: Rect,
) -> npt.NDArray[np.uint8]:
    """
    Draws an image onto a given scene with specified opacity and dimensions.

    Args:
        scene: Background image where the new image will be drawn.
        image: Image to draw, either a file path or an already-loaded image array.
        opacity: Opacity of the image to be drawn.
        rect: Rectangle specifying where to draw the image.

    Returns:
        The updated scene.

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

    rect_x = int(rect.x)
    rect_y = int(rect.y)
    rect_width = int(rect.width)
    rect_height = int(rect.height)
    # Validate rectangle dimensions
    if (
        rect_x < 0
        or rect_y < 0
        or rect_x + rect_width > scene.shape[1]
        or rect_y + rect_height > scene.shape[0]
    ):
        raise ValueError("Invalid rectangle dimensions.")

    # Resize and isolate alpha channel
    image = cv2.resize(image, (rect_width, rect_height))
    image = cast(npt.NDArray[np.uint8], image)
    alpha_channel = (
        image[:, :, 3]
        if image.shape[2] == 4
        else np.ones((rect_height, rect_width), dtype=image.dtype) * 255
    )
    alpha_scaled = cv2.convertScaleAbs(alpha_channel * opacity)

    # Perform blending
    scene_roi = scene[rect_y : rect_y + rect_height, rect_x : rect_x + rect_width]
    alpha_float = alpha_scaled.astype(np.float32) / 255.0
    blended_roi = cv2.convertScaleAbs(
        (1 - alpha_float[..., np.newaxis]) * scene_roi
        + alpha_float[..., np.newaxis] * image[:, :, :3]
    )

    # Update the scene
    scene[rect_y : rect_y + rect_height, rect_x : rect_x + rect_width] = blended_roi

    return scene


def calculate_optimal_text_scale(resolution_wh: tuple[int, int]) -> float:
    """
    Calculate optimal font scale based on image resolution. Adjusts font scale
    proportionally to the smallest dimension of the given image resolution for
    consistent readability.

    Args:
        resolution_wh: A tuple of `(width, height)` of the image in pixels.

    Returns:
        Recommended font scale factor.

    Examples:
        ```pycon
        >>> import supervision as sv
        >>> sv.calculate_optimal_text_scale((1920, 1080))
        1.08
        >>> sv.calculate_optimal_text_scale((640, 480))
        0.48

        ```
    """
    return min(resolution_wh) * 1e-3


def calculate_optimal_line_thickness(resolution_wh: tuple[int, int]) -> int:
    """
    Calculate optimal line thickness based on image resolution. Adjusts the line
    thickness for readability depending on the smallest dimension of the provided
    image resolution.

    Args:
        resolution_wh: A tuple of `(width, height)` of the image in pixels.

    Returns:
        Recommended line thickness in pixels.

    Examples:
        ```pycon
        >>> import supervision as sv
        >>> sv.calculate_optimal_line_thickness((1920, 1080))
        4
        >>> sv.calculate_optimal_line_thickness((640, 480))
        2

        ```
    """
    if min(resolution_wh) < 1080:
        return 2
    return 4
