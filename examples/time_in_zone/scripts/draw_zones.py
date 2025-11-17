from __future__ import annotations

import argparse
import json
import os
from typing import Any

import cv2
import numpy as np

import supervision as sv

KEY_ENTER = 13
KEY_NEWLINE = 10
KEY_ESCAPE = 27
KEY_QUIT = ord("q")
KEY_SAVE = ord("s")

THICKNESS = 2
COLORS = sv.ColorPalette.DEFAULT
WINDOW_NAME = "Draw Zones"
POLYGONS = [[]]

current_mouse_position: tuple[int, int] | None = None


def resolve_source(source_path: str) -> np.ndarray | None:
    if not os.path.exists(source_path):
        return None

    image = cv2.imread(source_path)
    if image is not None:
        return resize_to_fit_screen(image)

    frame_generator = sv.get_video_frames_generator(source_path=source_path)
    frame = next(frame_generator)
    return resize_to_fit_screen(frame)


def resize_to_fit_screen(
    image: np.ndarray, max_width: int = 1200, max_height: int = 800
) -> np.ndarray:
    """
    Resize image to fit screen while maintaining aspect ratio.

    Args:
        image: Input image
        max_width: Maximum width for display
        max_height: Maximum height for display

    Returns:
        Resized image
    """
    height, width = image.shape[:2]

    # Calculate scaling factor
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h, 1.0)  # Don't upscale if image is smaller

    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        print(
            f"Video resolution resized from {width}x{height} -> {new_width}x{new_height}"
        )
        return resized

    return image


def mouse_event(event: int, x: int, y: int, flags: int, param: Any) -> None:
    global current_mouse_position
    if event == cv2.EVENT_MOUSEMOVE:
        current_mouse_position = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        POLYGONS[-1].append((x, y))


def redraw(image: np.ndarray, original_image: np.ndarray) -> None:
    global POLYGONS, current_mouse_position
    image[:] = original_image.copy()
    for idx, polygon in enumerate(POLYGONS):
        color = (
            COLORS.by_idx(idx).as_bgr()
            if idx < len(POLYGONS) - 1
            else sv.Color.WHITE.as_bgr()
        )

        if len(polygon) > 1:
            for i in range(1, len(polygon)):
                cv2.line(
                    img=image,
                    pt1=polygon[i - 1],
                    pt2=polygon[i],
                    color=color,
                    thickness=THICKNESS,
                )
            if idx < len(POLYGONS) - 1:
                cv2.line(
                    img=image,
                    pt1=polygon[-1],
                    pt2=polygon[0],
                    color=color,
                    thickness=THICKNESS,
                )
        if idx == len(POLYGONS) - 1 and current_mouse_position is not None and polygon:
            cv2.line(
                img=image,
                pt1=polygon[-1],
                pt2=current_mouse_position,
                color=color,
                thickness=THICKNESS,
            )
    cv2.imshow(WINDOW_NAME, image)


def close_and_finalize_polygon(image: np.ndarray, original_image: np.ndarray) -> None:
    if len(POLYGONS[-1]) > 2:
        cv2.line(
            img=image,
            pt1=POLYGONS[-1][-1],
            pt2=POLYGONS[-1][0],
            color=COLORS.by_idx(0).as_bgr(),
            thickness=THICKNESS,
        )
    POLYGONS.append([])
    image[:] = original_image.copy()
    redraw_polygons(image)
    cv2.imshow(WINDOW_NAME, image)


def redraw_polygons(image: np.ndarray) -> None:
    for idx, polygon in enumerate(POLYGONS[:-1]):
        if len(polygon) > 1:
            color = COLORS.by_idx(idx).as_bgr()
            for i in range(len(polygon) - 1):
                cv2.line(
                    img=image,
                    pt1=polygon[i],
                    pt2=polygon[i + 1],
                    color=color,
                    thickness=THICKNESS,
                )
            cv2.line(
                img=image,
                pt1=polygon[-1],
                pt2=polygon[0],
                color=color,
                thickness=THICKNESS,
            )


def convert_coordinates_to_original(polygons, original_size, display_size):
    """
    Convert coordinates from display size back to original video size.

    Args:
        polygons: List of polygons with display coordinates
        original_size: (width, height) of original video
        display_size: (width, height) of display window

    Returns:
        List of polygons with original coordinates
    """
    orig_w, orig_h = original_size
    disp_w, disp_h = display_size

    scale_x = orig_w / disp_w
    scale_y = orig_h / disp_h

    converted_polygons = []
    for polygon in polygons:
        if polygon:  # Skip empty polygons
            converted_polygon = []
            for x, y in polygon:
                orig_x = int(x * scale_x)
                orig_y = int(y * scale_y)
                converted_polygon.append([orig_x, orig_y])
            converted_polygons.append(converted_polygon)

    return converted_polygons


def save_polygons_to_json(polygons, target_path, original_size=None, display_size=None):
    data_to_save = polygons if polygons[-1] else polygons[:-1]

    # Convert coordinates back to original size if needed
    if original_size and display_size:
        data_to_save = convert_coordinates_to_original(
            data_to_save, original_size, display_size
        )

    with open(target_path, "w") as f:
        json.dump(data_to_save, f)


def main(source_path: str, zone_configuration_path: str) -> None:
    global current_mouse_position
    original_image = resolve_source(source_path=source_path)
    if original_image is None:
        print("Failed to load source image.")
        return

    # Get original video dimensions
    cap = cv2.VideoCapture(source_path)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Get display dimensions
    display_height, display_width = original_image.shape[:2]

    print(f"Original video size: {original_width}x{original_height}")
    print(f"Display size: {display_width}x{display_height}")

    image = original_image.copy()
    cv2.imshow(WINDOW_NAME, image)
    cv2.setMouseCallback(WINDOW_NAME, mouse_event, image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == KEY_ENTER or key == KEY_NEWLINE:
            close_and_finalize_polygon(image, original_image)
        elif key == KEY_ESCAPE:
            POLYGONS[-1] = []
            current_mouse_position = None
        elif key == KEY_SAVE:
            save_polygons_to_json(
                POLYGONS,
                zone_configuration_path,
                (original_width, original_height),
                (display_width, display_height),
            )
            print(f"Polygons saved to {zone_configuration_path}")
            print("Coordinates converted to original video size.")
            break
        redraw(image, original_image)
        if key == KEY_QUIT:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactively draw polygons on images or video frames and save "
        "the annotations."
    )
    parser.add_argument(
        "--source_path",
        type=str,
        required=True,
        help="Path to the source image or video file for drawing polygons.",
    )
    parser.add_argument(
        "--zone_configuration_path",
        type=str,
        required=True,
        help="Path where the polygon annotations will be saved as a JSON file.",
    )
    arguments = parser.parse_args()
    main(
        source_path=arguments.source_path,
        zone_configuration_path=arguments.zone_configuration_path,
    )
