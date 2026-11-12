import os
from typing import cast

import numpy as np
import numpy.typing as npt

from supervision import _cv2 as cv2
from supervision.draw.color import Color
from supervision.geometry.core import (
    CoordinatesTransformation,
    Point,
    Rect,
    _transform_points,
)

#: Largest absolute pixel coordinate a mapped point may have and still be drawn: it
#: fits int32 for OpenCV and bounds OpenCV `fillPoly` work, which grows with the
#: coordinate magnitude; the NumPy fallback clips shapes to the canvas.
_MAX_DRAWING_COORDINATE = 2**20


def _map_to_drawing_coordinates(
    points: npt.NDArray[np.number],
    coord_transform: CoordinatesTransformation,
) -> npt.NDArray[np.float64] | None:
    """Map reference-frame `(N, 2)` points into the current frame for drawing.

    Returns the mapped points unrounded, or `None` when any of them is non-finite
    or beyond `_MAX_DRAWING_COORDINATE`, so the caller skips the whole shape
    instead of distorting it by dropping vertices. `points` is copied first, so a
    transform that edits its input in place cannot change the zone.

    Only those points are skipped. A homography can map points behind the camera
    to finite but mirrored coordinates, for example when it clamps a tiny `w`
    instead of returning NaN; this protocol cannot detect that, so a zone crossing
    the horizon may be drawn folded.

    Raises:
        ValueError: If `coord_transform` does not return one point per input point.
    """
    points_copy = np.array(points, dtype=np.float64)
    mapped = _transform_points(points_copy, coord_transform.abs_to_rel)
    # Checked unrounded so rounding cannot pull an out-of-range point inside.
    if not np.isfinite(mapped).all():
        return None
    if (np.abs(mapped) > _MAX_DRAWING_COORDINATE).any():
        return None
    return mapped


def _color_for_scene(color: Color, scene: npt.NDArray[np.uint8]) -> tuple[int, ...]:
    """Return `color` as the value OpenCV should draw on `scene`.

    OpenCV fills the alpha a BGR color lacks with 0, so whatever it draws on a BGRA
    scene, shaped ``(H, W, 4)``, becomes transparent. Such a scene takes the color with
    an opaque alpha instead.
    """
    if scene.ndim == 3 and scene.shape[2] == 4:
        return (*color.as_bgr(), 255)
    return color.as_bgr()


def draw_line(
    scene: npt.NDArray[np.uint8],
    start: Point,
    end: Point,
    color: Color = Color.ROBOFLOW,
    thickness: int = 2,
) -> npt.NDArray[np.uint8]:
    """Draws a line on a given scene.

    Args:
        scene: The scene on which the line will be drawn
        start: The starting point of the line
        end: The end point of the line
        color: The color of the line, defaults to Color.ROBOFLOW
        thickness: The thickness of the line

    Returns:
        The scene with the line drawn on it

    Example:
        ```pycon
        >>> import numpy as np
        >>> from supervision.draw.utils import draw_line
        >>> from supervision.draw.color import Color
        >>> from supervision.geometry.core import Point
        >>> scene = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> scene = draw_line(
        ...     scene, start=Point(x=10, y=10), end=Point(x=90, y=90), color=Color.RED
        ... )
        >>> scene.shape
        (100, 100, 3)

        ```
    """
    cv2.line(
        scene,
        start.as_xy_int_tuple(),
        end.as_xy_int_tuple(),
        _color_for_scene(color, scene),
        thickness=thickness,
    )
    return scene


def draw_rectangle(
    scene: npt.NDArray[np.uint8],
    rect: Rect,
    color: Color = Color.ROBOFLOW,
    thickness: int = 2,
) -> npt.NDArray[np.uint8]:
    """Draws a rectangle on an image.

    Args:
        scene: The scene on which the rectangle will be drawn
        rect: The rectangle to be drawn
        color: The color of the rectangle
        thickness: The thickness of the rectangle border

    Returns:
        The scene with the rectangle drawn on it

    Example:
        ```pycon
        >>> import numpy as np
        >>> from supervision.draw.utils import draw_rectangle
        >>> from supervision.draw.color import Color
        >>> from supervision.geometry.core import Rect
        >>> scene = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> rect = Rect(x=10, y=10, width=50, height=30)
        >>> scene = draw_rectangle(scene, rect, color=Color.RED)
        >>> scene.shape
        (100, 100, 3)

        ```
    """
    cv2.rectangle(
        scene,
        rect.top_left.as_xy_int_tuple(),
        rect.bottom_right.as_xy_int_tuple(),
        _color_for_scene(color, scene),
        thickness=thickness,
    )
    return scene


def draw_filled_rectangle(
    scene: npt.NDArray[np.uint8],
    rect: Rect,
    color: Color = Color.ROBOFLOW,
    opacity: float = 1,
) -> npt.NDArray[np.uint8]:
    """Draws a filled rectangle on an image.

    Args:
        scene: The scene on which the rectangle will be drawn
        rect: The rectangle to be drawn
        color: The color of the rectangle
        opacity: The opacity of rectangle when drawn on the scene.

    Returns:
        The scene with the rectangle drawn on it

    Example:
        ```pycon
        >>> import numpy as np
        >>> from supervision.draw.utils import draw_filled_rectangle
        >>> from supervision.draw.color import Color
        >>> from supervision.geometry.core import Rect
        >>> scene = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> rect = Rect(x=10, y=10, width=50, height=30)
        >>> scene = draw_filled_rectangle(scene, rect, color=Color.RED, opacity=0.5)
        >>> scene.shape
        (100, 100, 3)

        ```
    """
    if opacity == 1:
        cv2.rectangle(
            scene,
            rect.top_left.as_xy_int_tuple(),
            rect.bottom_right.as_xy_int_tuple(),
            _color_for_scene(color, scene),
            -1,
        )
    else:
        scene_with_annotations = scene.copy()
        cv2.rectangle(
            scene_with_annotations,
            rect.top_left.as_xy_int_tuple(),
            rect.bottom_right.as_xy_int_tuple(),
            _color_for_scene(color, scene),
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
    """Draws a rounded rectangle on an image.

    Args:
        scene: The image on which the rounded rectangle will be drawn.
        rect: The rectangle to be drawn.
        color: The color of the rounded rectangle.
        border_radius: The radius of the corner rounding in pixels. Values <= 0
            (or values clamped to 0 when the rectangle is too small) draw a
            plain filled rectangle with square corners. Note: previously,
            a negative value that remained negative after clamping would raise
            ``cv2.error``; it now draws square corners silently.

    Returns:
        The image with the rounded rectangle drawn on it.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from supervision.draw.utils import draw_rounded_rectangle
        >>> from supervision.draw.color import Color
        >>> from supervision.geometry.core import Rect
        >>> scene = np.zeros((200, 300, 3), dtype=np.uint8)
        >>> rect = Rect(x=20, y=30, width=120, height=80)
        >>> scene = draw_rounded_rectangle(scene, rect, Color.RED, border_radius=0)
        >>> scene.shape
        (200, 300, 3)

        ```
    """
    x1, y1, x2, y2 = rect.as_xyxy_int_tuple()
    width, height = x2 - x1, y2 - y1
    border_radius = min(border_radius, min(width, height) // 2)

    if border_radius <= 0:
        # square corners: a single fill rectangle (the common default), rather
        # than two rectangles plus four zero-radius corner circles
        cv2.rectangle(
            img=scene,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=_color_for_scene(color, scene),
            thickness=-1,
        )
        return scene

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
            color=_color_for_scene(color, scene),
            thickness=-1,
        )
    for center in circle_centers:
        cv2.circle(
            img=scene,
            center=center,
            radius=border_radius,
            color=_color_for_scene(color, scene),
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

    Example:
        ```pycon
        >>> import numpy as np
        >>> from supervision.draw.utils import draw_polygon
        >>> from supervision.draw.color import Color
        >>> scene = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> polygon = np.array([[10, 10], [90, 10], [90, 90], [10, 90]])
        >>> scene = draw_polygon(scene, polygon, color=Color.RED)
        >>> scene.shape
        (100, 100, 3)

        ```
    """
    cv2.polylines(
        scene,
        [polygon],
        isClosed=True,
        color=_color_for_scene(color, scene),
        thickness=thickness,
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

    Example:
        ```pycon
        >>> import numpy as np
        >>> from supervision.draw.utils import draw_filled_polygon
        >>> from supervision.draw.color import Color
        >>> scene = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> polygon = np.array([[10, 10], [90, 10], [90, 90], [10, 90]])
        >>> scene = draw_filled_polygon(scene, polygon, color=Color.RED, opacity=0.5)
        >>> scene.shape
        (100, 100, 3)

        ```
    """
    if opacity == 1:
        cv2.fillPoly(scene, [polygon], color=_color_for_scene(color, scene))
    else:
        scene_with_annotations = scene.copy()
        cv2.fillPoly(
            scene_with_annotations, [polygon], color=_color_for_scene(color, scene)
        )
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
    """Draw text with background on a scene.

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

    text_origin = (text_anchor_x - text_width // 2, text_anchor_y + text_height // 2)
    if scene.ndim == 3 and scene.shape[2] == 4:
        # OpenCV 5 writes glyph coverage into the alpha channel even for an opaque
        # color, so draw the colors and the alpha separately. A covered pixel gets
        # the color it has on the BGR copy and its alpha only ever rises.
        bgr = np.ascontiguousarray(scene[..., :3])
        coverage = np.zeros(scene.shape[:2], dtype=np.uint8)
        for target, target_color in ((bgr, text_color.as_bgr()), (coverage, (255,))):
            cv2.putText(
                img=target,
                text=text,
                org=text_origin,
                fontFace=text_font,
                fontScale=text_scale,
                color=target_color,
                thickness=text_thickness,
                lineType=cv2.LINE_AA,
            )
        scene[..., :3] = bgr
        scene[..., 3] = np.maximum(scene[..., 3], coverage)
        return scene

    cv2.putText(
        img=scene,
        text=text,
        org=text_origin,
        fontFace=text_font,
        fontScale=text_scale,
        color=text_color.as_bgr(),
        thickness=text_thickness,
        lineType=cv2.LINE_AA,
    )
    return scene


def _overlay_in_scene_channels(
    bgr: npt.NDArray[np.uint8], channels: int
) -> npt.NDArray[np.uint8]:
    """Return a BGR overlay with the channel count of the scene it is drawn on.

    One channel takes the overlay's gray luminance. Four channels append an opaque
    alpha, so blending makes the drawn pixels of a BGRA scene opaque in proportion to
    the overlay's alpha.
    """
    if channels == 1:
        gray = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2GRAY)
        return cast(npt.NDArray[np.uint8], gray)[..., np.newaxis]
    if channels == 4:
        opaque = np.full((*bgr.shape[:2], 1), 255, dtype=np.uint8)
        return np.concatenate((bgr, opaque), axis=2)
    return bgr


def _composite_over_bgra(
    scene: npt.NDArray[np.uint8],
    overlay: npt.NDArray[np.uint8],
    alpha: npt.NDArray[np.float32],
) -> npt.NDArray[np.uint8]:
    """Composite `overlay` over a straight-alpha BGRA region.

    Colors are weighted by how much each layer shows and divided by the result's
    alpha, so a translucent draw on a transparent scene keeps its color instead of
    fading toward the scene's hidden one. Where the result has no alpha at all the
    scene's color is kept.

    Args:
        scene: BGRA region of shape `(H, W, 4)`.
        overlay: BGR pixels of shape `(H, W, 3)`.
        alpha: Overlay coverage in `[0, 1]`, of shape `(H, W, 1)`.

    Returns:
        The composited BGRA region of shape `(H, W, 4)`.
    """
    scene_colors = scene[..., :3].astype(np.float32)
    scene_weight = (1 - alpha) * (scene[..., 3:].astype(np.float32) / 255.0)
    output_alpha = alpha + scene_weight
    colors = np.divide(
        alpha * overlay + scene_weight * scene_colors,
        output_alpha,
        out=scene_colors.copy(),
        where=np.broadcast_to(output_alpha > 0, scene_colors.shape),
    )
    return cast(
        npt.NDArray[np.uint8],
        cv2.convertScaleAbs(np.dstack([colors, output_alpha * 255.0])),
    )


def draw_image(
    scene: npt.NDArray[np.uint8],
    image: str | npt.NDArray[np.uint8],
    opacity: float,
    rect: Rect,
) -> npt.NDArray[np.uint8]:
    """Draws an image onto a given scene with specified opacity and dimensions.

    Args:
        scene: Background image where the new image will be drawn: BGR shaped
            ``(H, W, 3)``, BGRA shaped ``(H, W, 4)``, or grayscale shaped
            ``(H, W)`` or ``(H, W, 1)``. A grayscale scene receives the gray
            luminance of the image. On a BGRA scene, drawn pixels also become
            opaque in proportion to the image's alpha and `opacity`.
        image: Image to draw, either a file path or an already-loaded image array.
            A 2-D grayscale image is not accepted, whether it is passed as an array
            or read from a path; convert it to BGR first, for example with
            `cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)`. A 16-bit image read from a
            path is scaled down to the 8 bits of the scene.
        opacity: Opacity of the image to be drawn.
        rect: Rectangle specifying where to draw the image.

    Returns:
        The updated scene.

    Raises:
        FileNotFoundError: If the image path does not exist.
        OSError: If the image path exists but cannot be decoded.
        ValueError: If the image is not 3-D with 3 (BGR) or 4 (BGRA) channels, or
            for invalid opacity or rectangle dimensions.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from supervision.draw.utils import draw_image
        >>> from supervision.geometry.core import Rect
        >>> scene = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> image = np.full((40, 40, 3), 255, dtype=np.uint8)
        >>> rect = Rect(x=10, y=10, width=40, height=40)
        >>> scene = draw_image(scene, image, opacity=0.8, rect=rect)
        >>> scene.shape
        (100, 100, 3)

        ```
    """

    # Validate and load image
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image path ('{image}') does not exist.")
        loaded_image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if loaded_image is None:
            raise OSError(f"Could not decode image path ('{image}').")
        if loaded_image.dtype == np.uint16:
            # IMREAD_UNCHANGED keeps a 16-bit PNG at 16 bits, which the blend into
            # the 8-bit scene saturates to 255; keep the high byte instead.
            loaded_image = np.clip(np.rint(loaded_image / 256), 0, 255).astype(np.uint8)
        image_np = cast(npt.NDArray[np.uint8], loaded_image)
    else:
        image_np = image

    if image_np.ndim != 3 or image_np.shape[2] not in (3, 4):
        raise ValueError("Image must have 3 or 4 channels.")

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
    image_np = cast(
        npt.NDArray[np.uint8], cv2.resize(image_np, (rect_width, rect_height))
    )
    alpha_channel = (
        image_np[:, :, 3]
        if image_np.shape[2] == 4
        else np.ones((rect_height, rect_width), dtype=image_np.dtype) * 255
    )
    alpha_scaled = cv2.convertScaleAbs(alpha_channel * opacity)

    # Perform blending
    scene_roi = scene[rect_y : rect_y + rect_height, rect_x : rect_x + rect_width]
    scene_channels = scene_roi.reshape(rect_height, rect_width, -1)
    overlay = _overlay_in_scene_channels(image_np[:, :, :3], scene_channels.shape[2])
    alpha_float = alpha_scaled.astype(np.float32) / 255.0
    if scene_channels.shape[2] == 4:
        blended_roi = _composite_over_bgra(
            scene_channels, overlay[..., :3], alpha_float[..., np.newaxis]
        )
    else:
        blended_roi = cv2.convertScaleAbs(
            (1 - alpha_float[..., np.newaxis]) * scene_channels
            + alpha_float[..., np.newaxis] * overlay
        )

    # Update the scene; OpenCV drops the axis of a one-channel result.
    scene[rect_y : rect_y + rect_height, rect_x : rect_x + rect_width] = (
        blended_roi.reshape(scene_roi.shape)
    )

    return scene


def calculate_optimal_text_scale(resolution_wh: tuple[int, int]) -> float:
    """Calculate optimal font scale based on image resolution.

    Adjusts font scale proportionally to the smallest dimension of the given image
    resolution for consistent readability.

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
    """Calculate optimal line thickness based on image resolution.

    Adjusts the line thickness for readability depending on the smallest dimension of
    the provided image resolution.

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
